import { Grid, GridItem, Box, Heading } from "@chakra-ui/react"
import { SocketContext } from "@/contexts/SocketContext";
import { useContext } from "react"

export default function Webcam() {
    const { name, callAccepted, myVideo, userVideo, callEnded, stream, call } = useContext<any>(SocketContext);

    return (
        <Grid justifyContent="center" templateColumns='repeat(2, 1fr)' mt="12">
            {/* my video */}
            {
                stream && (
                    <Box>
                        <GridItem colSpan={1}>
                            <Heading as="h5">
                                {name || 'Name'}
                            </Heading>
                            <video playsInline muted ref={myVideo} autoPlay width="600" />
                        </GridItem>
                    </Box>
                )
            }
            {/* user's video */}
            {
                callAccepted && !callEnded && (
                    <Box>
                        <GridItem colSpan={1}>
                            <Heading as="h5">
                                {call.name || 'Name'}
                            </Heading>
                            <video playsInline ref={userVideo} autoPlay width="600" />
                        </GridItem>
                    </Box>
                )
            }
        </Grid>
    )
}
