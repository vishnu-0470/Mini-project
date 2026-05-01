/**
 * VoiceLink — Hardened Signaling Server
 *
 * Fixes over original:
 *  ✅ Room size limit (max 2 peers) — prevents unwanted join-bombing
 *  ✅ Per-socket rate limiting on 'signal' events
 *  ✅ Rooms auto-cleaned when all peers disconnect
 *  ✅ Signal payload validation (prevents XSS / malformed data relay)
 *  ✅ CORS origin list (replace * with your domain in production)
 *  ✅ Graceful handling of duplicate joins
 */

const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http, {
  cors: {
    // 🔒 Replace "*" with your actual frontend origin in production:
    // e.g. "https://yourapp.com"
    origin: process.env.ALLOWED_ORIGIN || '*',
    methods: ['GET', 'POST']
  },
  // Prevent oversized payloads (ICE candidates are small; large blobs = abuse)
  maxHttpBufferSize: 1e5 // 100 KB max per message
});

app.use(express.static(__dirname));
app.get('/', (req, res) => res.sendFile(__dirname + '/index.html'));

// ── Room State ────────────────────────────────────────────────────────────────
// Map<roomId, Set<socketId>>
const rooms = new Map();
const MAX_PEERS_PER_ROOM = 2;

// ── Rate Limiting ─────────────────────────────────────────────────────────────
// Map<socketId, { count, resetAt }>
const signalRateMap = new Map();
const RATE_LIMIT = 60;           // max signal events
const RATE_WINDOW_MS = 10_000;   // per 10-second window

function isRateLimited(socketId) {
  const now = Date.now();
  let entry = signalRateMap.get(socketId);

  if (!entry || now > entry.resetAt) {
    entry = { count: 0, resetAt: now + RATE_WINDOW_MS };
    signalRateMap.set(socketId, entry);
  }

  entry.count++;
  return entry.count > RATE_LIMIT;
}

// ── Signal Validation ─────────────────────────────────────────────────────────
// Only relay messages that look like valid WebRTC signaling payloads.
function isValidSignal(signal) {
  if (!signal || typeof signal !== 'object') return false;

  // SDP offer/answer
  if (signal.type === 'offer' || signal.type === 'answer') {
    return typeof signal.sdp === 'string' && signal.sdp.length < 20_000;
  }

  // ICE candidate
  if (signal.candidate !== undefined) {
    if (signal.candidate === null) return true; // end-of-candidates sentinel
    return (
      typeof signal.candidate === 'object' &&
      typeof signal.candidate.candidate === 'string'
    );
  }

  return false;
}

// ── Room ID Validation ────────────────────────────────────────────────────────
function isValidRoomId(id) {
  // Allow only alphanumeric + hyphen, 4–32 chars
  return typeof id === 'string' && /^[A-Za-z0-9\-]{4,32}$/.test(id);
}

// ── Socket Handlers ───────────────────────────────────────────────────────────
io.on('connection', (socket) => {
  console.log(`[+] Connected: ${socket.id}`);

  socket.on('join', (roomId) => {
    // 1. Validate room ID
    if (!isValidRoomId(roomId)) {
      socket.emit('error', { code: 'INVALID_ROOM', message: 'Invalid room ID format.' });
      return;
    }

    // 2. Enforce room capacity
    if (!rooms.has(roomId)) rooms.set(roomId, new Set());
    const room = rooms.get(roomId);

    if (room.size >= MAX_PEERS_PER_ROOM) {
      socket.emit('error', { code: 'ROOM_FULL', message: 'Room is full (max 2 peers).' });
      return;
    }

    // 3. Prevent duplicate joins (e.g. double-click)
    if (room.has(socket.id)) {
      socket.emit('error', { code: 'ALREADY_JOINED', message: 'Already in this room.' });
      return;
    }

    // 4. Join
    socket.join(roomId);
    room.add(socket.id);
    socket.data.room = roomId; // store for cleanup on disconnect

    console.log(`[~] ${socket.id} joined room "${roomId}" (${room.size}/${MAX_PEERS_PER_ROOM})`);

    // Notify existing peers that someone joined
    socket.to(roomId).emit('peer-joined', socket.id);

    // Confirm to the joining peer (useful for UI feedback)
    socket.emit('joined', { roomId, peerCount: room.size });
  });

  socket.on('signal', (data) => {
    // 1. Rate limit
    if (isRateLimited(socket.id)) {
      socket.emit('error', { code: 'RATE_LIMITED', message: 'Too many signals. Slow down.' });
      return;
    }

    // 2. Validate shape
    if (
      !data ||
      typeof data.roomId !== 'string' ||
      !isValidRoomId(data.roomId) ||
      !isValidSignal(data.signal)
    ) {
      socket.emit('error', { code: 'INVALID_SIGNAL', message: 'Malformed signal payload.' });
      return;
    }

    // 3. Confirm socket is actually in that room
    const room = rooms.get(data.roomId);
    if (!room || !room.has(socket.id)) {
      socket.emit('error', { code: 'NOT_IN_ROOM', message: 'You are not in that room.' });
      return;
    }

    // 4. Relay (never broadcast back to sender)
    socket.to(data.roomId).emit('signal', {
      sender: socket.id,
      signal: data.signal
    });
  });

  socket.on('disconnect', () => {
    const roomId = socket.data.room;
    if (roomId && rooms.has(roomId)) {
      const room = rooms.get(roomId);
      room.delete(socket.id);

      // Notify remaining peers so they can show "peer lost" UI
      socket.to(roomId).emit('peer-left', socket.id);

      // Clean up empty rooms — prevents unbounded Map growth
      if (room.size === 0) {
        rooms.delete(roomId);
        console.log(`[−] Room "${roomId}" deleted (empty).`);
      } else {
        console.log(`[−] ${socket.id} left room "${roomId}" (${room.size} remaining).`);
      }
    }

    signalRateMap.delete(socket.id);
    console.log(`[−] Disconnected: ${socket.id}`);
  });
});

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
http.listen(PORT, () => {
  console.log(`✅ VoiceLink Signaling Server running on port ${PORT}`);
});
const dgram = require('dgram');
const udpClient = dgram.createSocket('udp4');
const MATLAB_PORT = 5005; // The port MATLAB will listen on

// Inside your existing Socket.io connection block:
io.on('connection', (socket) => {
    console.log('🟢 User connected to Render cloud:', socket.id);

    // If you have WebRTC video/audio routing, keep it here:
    socket.on('join-room', (roomId) => {
        socket.join(roomId);
        socket.to(roomId).emit('user-connected', socket.id);
    });

    // RELAY TELEMETRY: 
    // Render receives the data from the browser, and instantly broadcasts it 
    // back out over the internet to your local-bridge.js
    socket.on('dsp-telemetry', (data) => {
        socket.broadcast.emit('live-data', data); 
    });

    socket.on('disconnect', () => {
        console.log('🔴 User disconnected:', socket.id);
    });
});