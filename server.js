const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http, { cors: { origin: "*" } });

// Serve static files (like your noise-engine.js)
app.use(express.static(__dirname));

// Explicitly serve the HTML file when someone visits the main link
app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

io.on('connection', (socket) => {
    console.log(`User connected: ${socket.id}`);

    socket.on('join', (roomId) => {
        socket.join(roomId);
        console.log(`User ${socket.id} joined room: ${roomId}`);
        socket.to(roomId).emit('peer-joined', socket.id);
    });

    socket.on('signal', (data) => {
        socket.to(data.roomId).emit('signal', {
            sender: socket.id,
            signal: data.signal
        });
    });

    socket.on('disconnect', () => {
        console.log(`User disconnected: ${socket.id}`);
    });
});

http.listen(3001, () => {
    console.log('VoiceLink Unified Server running on port 3001');
});