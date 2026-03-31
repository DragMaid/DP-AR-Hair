import { createContext, useState, useRef, useEffect } from 'react';
import { io } from 'socket.io-client';
import Peer, { Instance } from 'simple-peer';
import type { Call } from '@/types/callType';

const SocketContext = createContext<any>(undefined);
const socket = io('http://localhost:8080');  // TODO: Add proper address for future deployment
function SocketContextProvider({ children }: { children: React.ReactNode }) {
    const [callAccepted, setCallAccepted] = useState<boolean>(false);
    const [callEnded, setCallEnded] = useState<boolean>(false);
    const [stream, setStream] = useState<MediaStream>();
    const [name, setName] = useState<string>('');
    const [call, setCall] = useState<Call>({ isReceivingCall: false, from: '', name: '', signal: '' });
    const [me, setMe] = useState<string>('');
    const myVideo = useRef<HTMLVideoElement>(null);
    const userVideo = useRef<HTMLVideoElement>(null);
    const connectionRef = useRef<Instance | null>(null);

    // Gain access to user's webcam
    useEffect(() => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(currentStream => {
                setStream(currentStream);
                if (myVideo.current) {
                    myVideo.current.srcObject = currentStream;
                }
            })
        socket.on('me', id => setMe(id));
        socket.on('callUser', ({ from, name: callerName, signal }) => {
            setCall({ isReceivingCall: true, from, name: callerName, signal });
        });
    }, []);

    // User makes a call
    function callUser(id: string) {
        const peer = new Peer({ initiator: true, trickle: false, stream });
        peer.on('signal', data => {
            socket.emit('callUser', { userToCall: id, signalData: data, from: me, name });
        });
        peer.on('stream', currentStream => {
            if (userVideo.current) userVideo.current.srcObject = currentStream;
        })
        socket.on('callAccepted', signal => {
            setCallAccepted(true);
            peer.signal(signal);
        })
        connectionRef.current = peer;
    }

    // User answers a call
    function answerCall() {
        setCallAccepted(true);
        const peer = new Peer({ initiator: false, trickle: false, stream });
        peer.on('signal', data => {
            socket.emit('answerCall', { signal: data, to: call.from });
        });
        peer.on('stream', currentStream => {
            if (userVideo.current) userVideo.current.srcObject = currentStream;
        });
        peer.signal(call.signal);
        connectionRef.current = peer;
    }

    // User leaves the call
    function leaveCall() {
        setCallEnded(true);
        connectionRef.current && connectionRef.current.destroy();
        window.location.reload();
    }

    return (
        <SocketContext.Provider value={{
            call,
            callAccepted,
            myVideo,
            userVideo,
            stream,
            name,
            setName,
            callEnded,
            me,
            callUser,
            leaveCall,
            answerCall,
        }}
        >
            {children}
        </SocketContext.Provider>
    )
};

export { SocketContextProvider, SocketContext };
