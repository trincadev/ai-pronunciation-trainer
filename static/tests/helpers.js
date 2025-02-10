const mediaStreamConstraints = {
    audio: {
        channelCount: 1,
        sampleRate: 48000
    }
}

// https://github.com/bryanjenningz/record-audio/
export const audioRecorder = () =>
    new Promise(async resolve => {
        console.log("audioRecorder :: starting media device, navigator:", typeof navigator, ">", navigator, "#")
        let mmm = navigator.mediaDevices;
        console.log("audioRecorder :: starting media device, mediaDevices:", typeof mmm, ">", mmm, "#")
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        console.log("audioRecorder :: starting media device, stream:", typeof stream, stream, "#")
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", event => {
            audioChunks.push(event.data);
        });

        const start = () => mediaRecorder.start();

        const stop = () =>
            new Promise(resolve => {
                mediaRecorder.addEventListener("stop", () => {
                    const audioBlob = new Blob(audioChunks, { type: "audio/ogg" });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    const play = () => audio.play();
                    stream.getTracks().forEach((track) => track.stop());
                    resolve({ audioBlob, audioUrl, play });
                });

                mediaRecorder.stop();
            });

        resolve({ start, stop });
    });

const audioRecorderOld = async (audioChunks) => {
    navigator.mediaDevices.getUserMedia(mediaStreamConstraints).then(_stream => {
        stream = _stream
        mediaRecorder = new MediaRecorder(stream);

        let currentSamples = 0
        mediaRecorder.ondataavailable = event => {
            currentSamples += event.data.length
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/ogg;' });
            let audioUrl = URL.createObjectURL(audioBlob);
            audioRecorded = new Audio(audioUrl);
            let audioBase64 = await convertBlobToBase64(audioBlob);

            let minimumAllowedLength = 6;
            if (audioBase64.length < minimumAllowedLength) {
                setTimeout(UIRecordingError, 50); // Make sure this function finished after get called again
                return;
            }
            return audioBase64
        };
    });
}
