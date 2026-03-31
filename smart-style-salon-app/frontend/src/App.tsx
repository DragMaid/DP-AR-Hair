import React from 'react';
import { Box, Heading, Container } from '@chakra-ui/react';
import Webcam from './components/Webcam';
import ImageUploadBox from './components/ImageUploadBox';
import './App.css';

export default function App() {
  return (
    <Box>
      <Container maxW="1200px" mt="8">
        <Heading as="h2" size="3xl">AR Hairstyle Preview</Heading>
        <Webcam></Webcam>
        <ImageUploadBox></ImageUploadBox>
      </Container>
    </Box>
  );
}
