import express, { Request, Response } from "express";
import { createServer } from "http";
import cors from "cors";
import { Server } from "socket.io";
import multer from "multer";

const app = express();
const server = createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
})
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'hairstyle_image');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});
const upload = multer({ storage: storage }).single('file');

// Server init
app.use(cors());
app.use(express.static('hairstyle_image'));
const PORT = process.env.PORT || 8080;
app.get('/', (req: Request, res: Response) => {
    res.send(`Hello from ${PORT}`);
});


// POST: send image
app.post('/image', (req: Request, res: Response) => {
    console.log("Received image: ", req);

    upload(req, res, err => {
        if (err) res.sendStatus(500);
        res.send(req.file);
    });
});


// WebRTC Signal Server
io.on("connection", (socket) => {
    socket.emit("me", socket.id);

    // Client makes a call
    socket.on("callUser", ({ userToCall, signalData, from, name }) => {
        io.to(userToCall).emit("callUser", { signal: signalData, from, name });
    });

    // Client answers a call
    socket.on("answerCall", (data) => {
        io.to(data.to).emit("callAccepted", data.signal)
    });

    // Client disconnected
    socket.on("disconnect", () => {
        socket.broadcast.emit("callEnded")
    });
});

server.listen(PORT, () => console.log(`Server is running on port ${PORT}`));