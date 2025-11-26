import express, { Request, Response } from "express";
import { createServer } from "http";
import cors from "cors";
import { Server } from "socket.io";

const app = express();
const server = createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
})

app.use(cors());
const PORT = process.env.PORT || 8080;
app.get('/', (req: Request, res: Response) => {
    res.send(`Hello from ${PORT}`);
});

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