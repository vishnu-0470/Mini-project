/**
 * VoiceLink Phase 4 — Hybrid TinyML / DSP Noise Engine
 *
 * Architecture (RNNoise-inspired):
 *  1.  STFT (FFT-512, 75% Hann overlap-add)        ← unchanged
 *  2.  Minimum Statistics Noise Estimator          ← unchanged
 *  3.  Decision-Directed Wiener Filter             ← unchanged, gains now soft
 *  4.  ★ TinyGRU Neural VAD (replaces SFM heuristic)
 *  5.  Post-OLA Soft-Knee Expander (gain = speechProb directly) ← cleaner
 *  6.  Online Welford Enrollment                   ← unchanged
 */

class NoiseEngine {
  constructor() {
    this.audioCtx      = null;
    this.sourceNode    = null;
    this.analyser      = null;
    this.workletNode   = null;
    this.destNode      = null;
    this.voiceprint    = null;
    this.voiceprintStd = null;
    this._blobUrl      = null;
    this._enrollTimer  = null;
    this.isEnrolling   = false;
    this.isEnabled     = false;
    this.maskStrength  = 2.5;

    // ★ NEW: Callback to send data to Render/MATLAB
    this.onTelemetry   = null; 

    this._metrics = {
      frameEnergy:    0,
      gain:           1,
      speechProb:     0,
      voiceMatchScore:0,
      snrDb:          0,
      isVoiced:       false,
      sfm:            1.0
    };
  }

  async init(rawMicStream) {
    if (this.audioCtx) await this.destroy();

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate:  48000,
      latencyHint: 'interactive'
    });

    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }

    const workletCode = `
      // ══════════════════════════════════════════════════════════════════════
      // Cooley-Tukey Radix-2 In-Place FFT  (unchanged)
      // ══════════════════════════════════════════════════════════════════════
      function fft(re, im) {
        const N = re.length;
        for (let i = 1, j = 0; i < N; i++) {
          let bit = N >> 1;
          for (; j & bit; bit >>= 1) j ^= bit;
          j ^= bit;
          if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
          }
        }
        for (let len = 2; len <= N; len <<= 1) {
          const ang = -2 * Math.PI / len;
          const wRe = Math.cos(ang), wIm = Math.sin(ang);
          for (let i = 0; i < N; i += len) {
            let curRe = 1, curIm = 0;
            const half = len >> 1;
            for (let j = 0; j < half; j++) {
              const uRe = re[i+j],      uIm = im[i+j];
              const vRe = re[i+j+half]*curRe - im[i+j+half]*curIm;
              const vIm = re[i+j+half]*curIm + im[i+j+half]*curRe;
              re[i+j]      = uRe + vRe;  im[i+j]      = uIm + vIm;
              re[i+j+half] = uRe - vRe;  im[i+j+half] = uIm - vIm;
              const nRe = curRe*wRe - curIm*wIm;
              curIm = curRe*wIm + curIm*wRe;
              curRe = nRe;
            }
          }
        }
      }

      function ifft(re, im) {
        const N = re.length;
        for (let i = 0; i < N; i++) im[i] = -im[i];
        fft(re, im);
        for (let i = 0; i < N; i++) { re[i] /= N; im[i] = -im[i] / N; }
      }

      // ══════════════════════════════════════════════════════════════════════
      // TinyGRU Neural VAD Constants
      // ══════════════════════════════════════════════════════════════════════
      const GRU_IN = 9;
      const GRU_H  = 8;

      const GRU_MEL_LO = new Uint8Array([ 1,  2,  5, 10, 20, 36]);
      const GRU_MEL_HI = new Uint8Array([ 2,  5, 10, 20, 36, 65]);

      const GRU_Wz = new Float32Array([
          0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.10, 0.25, 0.20,
          0.10, 0.15, 0.25, 0.25, 0.20, 0.10, 0.10, 0.25, 0.20,
          0.05, 0.05, 0.05, 0.10, 0.20, 0.25, 0.25, 0.10, 0.15,
          0.05, 0.10, 0.15, 0.20, 0.15, 0.05, 0.05, 0.30, 0.20,
          0.10, 0.10, 0.15, 0.20, 0.20, 0.10, 0.05, 0.20, 0.30,
          0.15, 0.25, 0.20, 0.15, 0.05, 0.05, 0.05, 0.20, 0.15,
          0.05, 0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.15, 0.20,
          0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.10, 0.20, 0.20,
      ]);

      const GRU_Uz = new Float32Array([
        0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15,
      ]);

      const GRU_bz = new Float32Array([-1.00, -1.00, -1.50, -1.00, -1.00, -1.00, -1.00, -1.00]);

      const GRU_Wr = new Float32Array([
          0.05, 0.10, 0.15, 0.15, 0.10, 0.05, 0.05, 0.20, 0.15,
          0.05, 0.10, 0.20, 0.20, 0.15, 0.05, 0.05, 0.20, 0.15,
          0.05, 0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.05, 0.10,
          0.05, 0.10, 0.10, 0.15, 0.10, 0.05, 0.05, 0.25, 0.15,
          0.05, 0.05, 0.10, 0.15, 0.15, 0.05, 0.05, 0.15, 0.25,
          0.10, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.15, 0.10,
          0.05, 0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.10, 0.15,
          0.05, 0.10, 0.15, 0.15, 0.10, 0.05, 0.05, 0.15, 0.15,
      ]);

      const GRU_Ur = new Float32Array([
        0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10,
      ]);

      const GRU_br = new Float32Array([0.50, 0.50, 0.30, 0.50, 0.40, 0.50, 0.40, 0.50]);

      const GRU_Wg = new Float32Array([
          0.00, 0.10, 1.20, 1.50, 0.50, 0.00, 0.10, 1.50, 0.50, 
          0.00, 0.20, 0.80, 1.00, 1.00, 0.20, 0.20, 1.20, 0.60, 
          0.00, 0.00, 0.00, 0.20, 1.00, 1.20, 1.50, 0.20, 0.30, 
          0.00, 0.30, 0.50, 0.80, 0.30, 0.00, 0.00, 2.00, 0.30, 
          0.00, 0.00, 0.50, 0.50, 0.50, 0.00, 0.00, 0.50, 1.80, 
          0.20, 0.80, 1.00, 0.50, 0.00, 0.00, 0.00, 1.00, 0.40, 
          0.00, 0.00, 0.30, 1.00, 1.20, 0.50, 0.50, 0.80, 0.50, 
          0.10, 0.20, 0.50, 0.80, 0.50, 0.20, 0.30, 1.00, 0.70, 
      ]);

      const GRU_Ug = new Float32Array([
        0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05,
        0.00, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05,
        0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.00, 0.05,
        0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.05,
        0.00, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.05,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.00, 0.05,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.05,
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.30,
      ]);

      const GRU_bg = new Float32Array([-1.50, -1.50, -2.00, -1.50, -1.50, -1.50, -1.50, -1.50]);
      const GRU_Wo = new Float32Array([0.80, 0.80, 0.50, 0.90, 0.60, 0.60, 0.70, 0.80]);
      const GRU_bo = -2.50; 

      // ══════════════════════════════════════════════════════════════════════
      // NoiseProcessor  (Phase 4 — TinyGRU Neural VAD)
      // ══════════════════════════════════════════════════════════════════════
      class NoiseProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          this.N    = 512;
          this.H    = 128;
          this.BINS = this.N / 2 + 1;

          this.win = new Float32Array(this.N);
          for (let i = 0; i < this.N; i++)
            this.win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.N - 1)));

          this.inBuf  = new Float32Array(this.N);
          this.outBuf = new Float32Array(this.N);
          this.re = new Float32Array(this.N);
          this.im = new Float32Array(this.N);
          this.curPow = new Float32Array(this.BINS);

          this.smoothPow = new Float32Array(this.BINS).fill(1e-8);
          this.MS        = 30;
          this.msBuf     = Array.from({ length: this.MS }, () => new Float32Array(this.BINS).fill(1e-8));
          this.msIdx     = 0;
          this.noisePow  = new Float32Array(this.BINS).fill(1e-8);
          this.MS_BIAS   = 1.66;

          this.DD_ALPHA      = 0.92;
          this.prevGain      = new Float32Array(this.BINS).fill(1.0);
          this.prevSpeechPow = new Float32Array(this.BINS).fill(1e-8);

          this.timeNoiseFloor = 0.001;
          this.expanderGain   = 1.0;

          this.isEnabled    = false;
          this.maskStrength = 2.5;
          this.voiceprint   = null;
          this.voiceLow     = 8;
          this.voiceHigh    = 72;

          this.gruH    = new Float32Array(GRU_H);   
          this.gruFeat = new Float32Array(GRU_IN);  
          this._gz     = new Float32Array(GRU_H);   
          this._gr     = new Float32Array(GRU_H);   
          this._gg     = new Float32Array(GRU_H);   
          this._grh    = new Float32Array(GRU_H);   

          this.port.onmessage = ({ data }) => {
            switch (data.type) {
              case 'SET_VOICEPRINT':
                this.voiceprint = data.payload;
                this.voiceLow   = data.voiceLow;
                this.voiceHigh  = data.voiceHigh;
                if (this.voiceprint) {
                  for (let k = 0; k < this.BINS && k < this.voiceprint.length; k++) {
                    if (k < this.voiceLow || k > this.voiceHigh) {
                      const p = Math.max(this.voiceprint[k] * this.voiceprint[k], 1e-9);
                      this.noisePow[k]      = p;
                      this.smoothPow[k]     = p;
                      this.prevSpeechPow[k] = p;
                      for (let f = 0; f < this.MS; f++) this.msBuf[f][k] = p;
                    }
                  }
                }
                break;
              case 'SET_ENABLED':      this.isEnabled    = data.payload; break;
              case 'SET_STRENGTH':     this.maskStrength = data.payload; break;
              case 'SET_GRU_WEIGHTS':
                const w = data.payload;
                if (w.Wz) GRU_Wz.set(w.Wz); if (w.Uz) GRU_Uz.set(w.Uz); if (w.bz) GRU_bz.set(w.bz);
                if (w.Wr) GRU_Wr.set(w.Wr); if (w.Ur) GRU_Ur.set(w.Ur); if (w.br) GRU_br.set(w.br);
                if (w.Wg) GRU_Wg.set(w.Wg); if (w.Ug) GRU_Ug.set(w.Ug); if (w.bg) GRU_bg.set(w.bg);
                if (w.Wo) GRU_Wo.set(w.Wo);
                this.gruH.fill(0);
                break;
            }
          };
        }

        static get parameterDescriptors() { return []; }

        _gruStep() {
          const x  = this.gruFeat;
          const h  = this.gruH;
          const z  = this._gz;
          const r  = this._gr;
          const g  = this._gg;
          const rh = this._grh;

          for (let i = 0; i < GRU_H; i++) {
            let s = GRU_bz[i];
            const wi = i * GRU_IN, ui = i * GRU_H;
            for (let j = 0; j < GRU_IN; j++) s += GRU_Wz[wi + j] * x[j];
            for (let j = 0; j < GRU_H;  j++) s += GRU_Uz[ui + j] * h[j];
            z[i] = 1.0 / (1.0 + Math.exp(-s));
          }

          for (let i = 0; i < GRU_H; i++) {
            let s = GRU_br[i];
            const wi = i * GRU_IN, ui = i * GRU_H;
            for (let j = 0; j < GRU_IN; j++) s += GRU_Wr[wi + j] * x[j];
            for (let j = 0; j < GRU_H;  j++) s += GRU_Ur[ui + j] * h[j];
            r[i] = 1.0 / (1.0 + Math.exp(-s));
          }

          for (let j = 0; j < GRU_H; j++) rh[j] = r[j] * h[j];
          for (let i = 0; i < GRU_H; i++) {
            let s = GRU_bg[i];
            const wi = i * GRU_IN, ui = i * GRU_H;
            for (let j = 0; j < GRU_IN; j++) s += GRU_Wg[wi + j] * x[j];
            for (let j = 0; j < GRU_H;  j++) s += GRU_Ug[ui + j] * rh[j];
            g[i] = Math.tanh(s);
          }

          for (let i = 0; i < GRU_H; i++)
            h[i] = (1.0 - z[i]) * h[i] + z[i] * g[i];

          let o = GRU_bo;
          for (let i = 0; i < GRU_H; i++) o += GRU_Wo[i] * h[i];
          return 1.0 / (1.0 + Math.exp(-o));
        }

        process(inputs, outputs) {
          const inp = inputs[0], out = outputs[0];
          if (!inp?.length || !out?.length) return true;

          const inFrame  = inp[0];
          const outFrame = out[0];

          if (!this.isEnabled) { outFrame.set(inFrame); return true; }

          let energy = 0;
          let zcr = 0;
          let prevSign = inFrame[0] >= 0 ? 1 : -1;

          for (let i = 0; i < inFrame.length; i++) {
            const sample = inFrame[i];
            energy += sample * sample;
            const sign = sample >= 0 ? 1 : -1;
            if (sign !== prevSign) { zcr++; prevSign = sign; }
          }

          energy = Math.sqrt(energy / inFrame.length);
          const zcrRate = zcr / inFrame.length;

          if (energy < this.timeNoiseFloor) {
            this.timeNoiseFloor = 0.90 * this.timeNoiseFloor + 0.10 * energy;
          } else {
            this.timeNoiseFloor = 0.999 * this.timeNoiseFloor + 0.001 * energy;
          }
          this.timeNoiseFloor = Math.max(0.0001, Math.min(this.timeNoiseFloor, 0.02));

          const globalSnr = energy / this.timeNoiseFloor;

          this.inBuf.copyWithin(0, this.H);
          this.inBuf.set(inFrame, this.N - this.H);

          for (let i = 0; i < this.N; i++) {
            this.re[i] = this.inBuf[i] * this.win[i];
            this.im[i] = 0;
          }

          fft(this.re, this.im);

          for (let k = 0; k < this.BINS; k++)
            this.curPow[k] = this.re[k]*this.re[k] + this.im[k]*this.im[k];

          const A = 0.85;
          for (let k = 0; k < this.BINS; k++)
            this.smoothPow[k] = A * this.smoothPow[k] + (1 - A) * this.curPow[k];

          this.msBuf[this.msIdx].set(this.smoothPow);
          this.msIdx = (this.msIdx + 1) % this.MS;

          this.noisePow.fill(Infinity);
          for (let f = 0; f < this.MS; f++) {
            const snap = this.msBuf[f];
            for (let k = 0; k < this.BINS; k++)
              if (snap[k] < this.noisePow[k]) this.noisePow[k] = snap[k];
          }
          for (let k = 0; k < this.BINS; k++)
            this.noisePow[k] = Math.max(this.noisePow[k] * this.MS_BIAS, 1e-10);

          const vlo = this.voiceLow, vhi = this.voiceHigh;
          const vCount = vhi - vlo + 1;
          let logSum = 0, linSum = 0;
          for (let k = vlo; k <= vhi; k++) {
            const p = Math.max(this.curPow[k], 1e-12);
            logSum += Math.log(p);
            linSum += p;
          }
          const geoMean   = Math.exp(logSum / vCount);
          const arithMean = linSum / vCount;
          const sfm       = geoMean / Math.max(arithMean, 1e-15);

          for (let b = 0; b < 6; b++) {
            let sum = 0;
            const lo = GRU_MEL_LO[b], hi = GRU_MEL_HI[b];
            for (let k = lo; k < hi; k++) sum += this.curPow[k];
            const logE = Math.log10(sum / (hi - lo) + 1e-10);
            this.gruFeat[b] = Math.max(0.0, Math.min(1.0, (logE + 10.0) / 10.0));
          }
          this.gruFeat[6] = Math.min(zcrRate * 4.0, 1.0);                             
          this.gruFeat[7] = Math.max(0.0, 1.0 - sfm / 0.15);                          
          this.gruFeat[8] = Math.min(Math.log(Math.max(globalSnr, 1.0)) / Math.log(20.0), 1.0); 

          const speechProb = this._gruStep();

          const aggr = this.maskStrength / 2.5;
          let gainSum   = 0;
          let snrLinSum = 0;

          for (let k = 0; k < this.BINS; k++) {
            const lN    = this.noisePow[k];
            const gamma = this.curPow[k] / lN;

            const xiRec  = this.DD_ALPHA * (this.prevGain[k] * this.prevGain[k] * this.prevSpeechPow[k]) / lN;
            const xiStep = (1 - this.DD_ALPHA) * Math.max(gamma - 1, 0);
            const xi     = xiRec + xiStep;

            let g = xi / (xi + 1.0);

            if (k >= vlo && k <= vhi) {
              const floor = speechProb * 0.45;
              g = g * (1.0 - floor) + floor;
            } else if (speechProb > 0.30 && k >= (vhi >> 1)) {
              const floor = (speechProb - 0.30) * 0.45;
              g = g * (1.0 - floor) + floor;
            }

            g = Math.pow(Math.max(g, 1e-6), aggr);
            if (g < 0.03) g = 0.03;
            if (g > 1.00) g = 1.00;

            this.prevGain[k]      = g;
            this.prevSpeechPow[k] = g * g * this.curPow[k];

            this.re[k] *= g;
            this.im[k] *= g;
            if (k > 0 && k < this.BINS - 1) {
              this.re[this.N - k] *= g;
              this.im[this.N - k] *= g;
            }

            gainSum   += g;
            snrLinSum += xi;
          }

          const avgGain = gainSum   / this.BINS;
          const avgXi   = snrLinSum / this.BINS;
          const snrDb   = 10 * Math.log10(Math.max(avgXi, 1e-6));

          ifft(this.re, this.im);

          for (let i = 0; i < this.N; i++)
            this.outBuf[i] += this.re[i];

          const targetGain = Math.max(0.05, speechProb);

          const alpha = targetGain > this.expanderGain ? 0.40 : 0.02;
          this.expanderGain = (1 - alpha) * this.expanderGain + alpha * targetGain;

          for (let j = 0; j < this.H; j++) {
            outFrame[j] = (this.outBuf[j] * 0.5) * this.expanderGain;
          }

          this.outBuf.copyWithin(0, this.H);
          this.outBuf.fill(0, this.N - this.H);

          this.port.postMessage({
            type:            'METRICS',
            frameEnergy:      energy,
            gain:             avgGain * this.expanderGain,
            snrDb,
            speechProb,                                   
            isVoiced:         speechProb > 0.45,  
            sfm,
            voiceMatchScore:  speechProb          
          });

          return true;
        }
      }

      registerProcessor('noise-processor', NoiseProcessor);
    `;

    const blob    = new Blob([workletCode], { type: 'application/javascript' });
    this._blobUrl = URL.createObjectURL(blob);
    await this.audioCtx.audioWorklet.addModule(this._blobUrl);
    URL.revokeObjectURL(this._blobUrl);
    this._blobUrl = null;

    this.sourceNode = this.audioCtx.createMediaStreamSource(rawMicStream);

    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize               = 2048;
    this.analyser.smoothingTimeConstant = 0.5;
    this.sourceNode.connect(this.analyser);

    this.workletNode = new AudioWorkletNode(this.audioCtx, 'noise-processor', {
      numberOfInputs:     1,
      numberOfOutputs:    1,
      outputChannelCount: [1]
    });
    this.destNode = this.audioCtx.createMediaStreamDestination();

    this.sourceNode.connect(this.workletNode);
    this.workletNode.connect(this.destNode);

    // ★ NEW: Catch the telemetry and fire the callback!
    this.workletNode.port.onmessage = ({ data }) => {
      if (data.type === 'METRICS') {
        this._metrics = data;
        if (this.onTelemetry) {
          this.onTelemetry(data);
        }
      }
    };

    console.log(
      '🟢 VoiceLink Phase 4 — Hybrid TinyML/DSP Pipeline ready\n' +
      '   → STFT (FFT-512) + Min-Stats + DD-Wiener\n' +
      '   → ★ TinyGRU Neural VAD (9 features → GRU-8 → speechProb)\n' +
      '   → Soft-Knee Expander (gate = speechProb directly)'
    );
    return this.destNode.stream;
  }

  // ── Enroll voiceprint (unchanged) ──────────────────────────────────────────
  async enrollVoice(durationMs = 4000) {
    if (!this.analyser) throw new Error('Engine not initialized — call init() first.');

    if (this._enrollTimer) {
      clearTimeout(this._enrollTimer);
      this._enrollTimer = null;
    }
    this.isEnrolling = false;

    await new Promise(r => setTimeout(r, 80));
    this.isEnrolling = true;
    console.log(`🎙️ Enrolling voiceprint for ${durationMs / 1000}s — speak now.`);

    const binCount = this.analyser.frequencyBinCount;
    const mean     = new Float32Array(binCount);
    const M2       = new Float32Array(binCount);
    let   n        = 0;

    return new Promise(resolve => {
      const capture = () => {
        if (!this.isEnrolling) return;
        const data = new Float32Array(binCount);
        this.analyser.getFloatFrequencyData(data);
        n++;
        for (let i = 0; i < binCount; i++) {
          const amp   = Math.pow(10, Math.max(data[i], -160) / 20);
          const delta = amp - mean[i];
          mean[i]    += delta / n;
          M2[i]      += delta * (amp - mean[i]);
        }
        requestAnimationFrame(capture);
      };
      capture();

      this._enrollTimer = setTimeout(() => {
        this.isEnrolling  = false;
        this._enrollTimer = null;

        this.voiceprint    = mean;
        this.voiceprintStd = new Float32Array(binCount);
        for (let i = 0; i < binCount; i++)
          this.voiceprintStd[i] = n > 1 ? Math.sqrt(M2[i] / (n - 1)) : 0;

        const nyquist   = this.audioCtx.sampleRate / 2;
        const binHz     = nyquist / binCount;
        const voiceLow  = Math.floor(300  / binHz);
        const voiceHigh = Math.floor(3400 / binHz);

        this.workletNode.port.postMessage({
          type:     'SET_VOICEPRINT',
          payload:  this.voiceprint,
          voiceLow,
          voiceHigh
        });

        console.log(
          `✅ Voiceprint captured. ${n} frames over ${durationMs}ms. ` +
          `Speech band: bins ${voiceLow}–${voiceHigh} (300–3400 Hz)`
        );
        resolve({ mean: this.voiceprint, std: this.voiceprintStd });
      }, durationMs);
    });
  }

  setGruWeights(weights) {
    if (!this.workletNode) throw new Error('Engine not initialized — call init() first.');
    this.workletNode.port.postMessage({ type: 'SET_GRU_WEIGHTS', payload: weights });
  }

  setEnabled(enabled) {
    this.isEnabled = enabled;
    this.workletNode?.port.postMessage({ type: 'SET_ENABLED', payload: enabled });
  }

  setStrength(value) {
    this.maskStrength = value;
    this.workletNode?.port.postMessage({ type: 'SET_STRENGTH', payload: value });
  }

  getMetrics() {
    return { ...this._metrics };
  }

  getVoiceprint() {
    if (!this.voiceprint) return null;
    return { mean: this.voiceprint, std: this.voiceprintStd };
  }

  async destroy() {
    if (this._enrollTimer) {
      clearTimeout(this._enrollTimer);
      this._enrollTimer = null;
    }
    this.isEnrolling = false;

    try { this.sourceNode?.disconnect();  } catch (_) {}
    try { this.workletNode?.disconnect(); } catch (_) {}

    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      await this.audioCtx.close();
    }

    this.audioCtx      = null;
    this.sourceNode    = null;
    this.analyser      = null;
    this.workletNode   = null;
    this.destNode      = null;
    this.voiceprint    = null;
    this.voiceprintStd = null;
    this.isEnabled     = false;

    console.log('🔴 DSP Pipeline destroyed.');
  }
}

window.VoiceLinkNoiseEngine = new NoiseEngine();