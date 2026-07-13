import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  AudioModule,
  RecordingPresets,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from 'expo-audio';
import * as Speech from 'expo-speech';

const API_URL = process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '');

type AskResponse = {
  transcript: string;
  answer: string;
  used_camera: boolean;
};

type SendOptions = {
  audioUri?: string;
  questionText?: string;
};

export default function App() {
  const cameraRef = useRef<CameraView | null>(null);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(audioRecorder);

  const [microphoneReady, setMicrophoneReady] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [includeCamera, setIncludeCamera] = useState(true);
  const [typedQuestion, setTypedQuestion] = useState('');
  const [transcript, setTranscript] = useState('');
  const [answer, setAnswer] = useState('');
  const [status, setStatus] = useState('Ready');
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    const prepareMicrophone = async () => {
      const permission = await AudioModule.requestRecordingPermissionsAsync();
      setMicrophoneReady(permission.granted);

      if (!permission.granted) {
        Alert.alert(
          'Microphone permission needed',
          'BELT needs microphone access for voice questions. You can still type a question instead.'
        );
      }

      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });
    };

    prepareMicrophone().catch((error) => {
      console.error(error);
      setStatus('Could not prepare the microphone.');
    });

    return () => {
      Speech.stop();
    };
  }, []);

  const capturePhoto = async (): Promise<string | undefined> => {
    if (!includeCamera || !cameraPermission?.granted || !cameraReady) {
      return undefined;
    }

    const photo = await cameraRef.current?.takePictureAsync({
      quality: 0.45,
      skipProcessing: false,
    });

    return photo?.uri;
  };

  const speakAnswer = async (text: string) => {
    await Speech.stop();
    Speech.speak(text, {
      language: 'en-US',
      rate: 0.95,
      pitch: 1.0,
    });
  };

  const sendToBELT = async ({ audioUri, questionText }: SendOptions) => {
    if (!API_URL || API_URL.includes('YOUR_MAC_IP')) {
      Alert.alert(
        'Backend address missing',
        'Open frontend/.env and replace YOUR_MAC_IP with your Mac’s local IP address.'
      );
      return;
    }

    setIsSending(true);
    setStatus('Taking a picture and asking BELT…');

    try {
      const imageUri = await capturePhoto();
      const form = new FormData();

      if (audioUri) {
        form.append(
          'audio',
          {
            uri: audioUri,
            name: 'question.m4a',
            type: 'audio/m4a',
          } as any
        );
      }

      if (imageUri) {
        form.append(
          'image',
          {
            uri: imageUri,
            name: 'surroundings.jpg',
            type: 'image/jpeg',
          } as any
        );
      }

      if (questionText?.trim()) {
        form.append('question_text', questionText.trim());
      }

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 60_000);

      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });

      clearTimeout(timeout);

      const data = (await response.json()) as AskResponse | { detail?: string };

      if (!response.ok) {
        const message = 'detail' in data && data.detail ? data.detail : 'BELT could not answer.';
        throw new Error(message);
      }

      const result = data as AskResponse;
      setTranscript(result.transcript || questionText || 'Camera-only question');
      setAnswer(result.answer);
      setStatus(result.used_camera ? 'Answered using your voice and camera.' : 'Answered without a camera image.');
      await speakAnswer(result.answer);
    } catch (error) {
      console.error(error);
      const message =
        error instanceof Error && error.name === 'AbortError'
          ? 'The request timed out. Check that the backend is running.'
          : error instanceof Error
            ? error.message
            : 'Something went wrong.';
      setStatus(message);
      Alert.alert('BELT error', message);
    } finally {
      setIsSending(false);
    }
  };

  const startRecording = async () => {
    if (!microphoneReady) {
      Alert.alert('Microphone unavailable', 'Allow microphone access or type your question below.');
      return;
    }

    try {
      await Speech.stop();
      setTranscript('');
      setAnswer('');
      await setAudioModeAsync({
        allowsRecording: true,
        playsInSilentMode: true,
      });
      await audioRecorder.prepareToRecordAsync();
      audioRecorder.record();
      setStatus('Listening… tap again when you finish speaking.');
    } catch (error) {
      console.error(error);
      setStatus('Could not start recording.');
    }
  };

  const stopRecordingAndAsk = async () => {
    try {
      setStatus('Finishing your recording…');
      await audioRecorder.stop();
      const audioUri = audioRecorder.uri ?? undefined;

      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });

      if (!audioUri) {
        throw new Error('The recording file was not created. Please try again.');
      }

      await sendToBELT({ audioUri });
    } catch (error) {
      console.error(error);
      const message = error instanceof Error ? error.message : 'Could not finish recording.';
      setStatus(message);
      Alert.alert('Recording error', message);
      setIsSending(false);
    }
  };

  const handleTalkButton = async () => {
    if (isSending) return;

    if (recorderState.isRecording) {
      await stopRecordingAndAsk();
    } else {
      await startRecording();
    }
  };

  const handleTypedQuestion = async () => {
    const question = typedQuestion.trim();
    if (!question) {
      Alert.alert('Type a question', 'Example: “Where is the nearest restroom?”');
      return;
    }

    await setAudioModeAsync({
      allowsRecording: false,
      playsInSilentMode: true,
    });
    await sendToBELT({ questionText: question });
  };

  const cameraContent = () => {
    if (!cameraPermission) {
      return (
        <View style={styles.cameraMessage}>
          <ActivityIndicator />
          <Text style={styles.cameraMessageText}>Checking camera permission…</Text>
        </View>
      );
    }

    if (!cameraPermission.granted) {
      return (
        <View style={styles.cameraMessage}>
          <Text style={styles.cameraMessageText}>
            Camera permission lets BELT look at signs and nearby landmarks.
          </Text>
          <Pressable style={styles.smallButton} onPress={requestCameraPermission}>
            <Text style={styles.smallButtonText}>Allow camera</Text>
          </Pressable>
        </View>
      );
    }

    return (
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
        onCameraReady={() => setCameraReady(true)}
      />
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView
        style={styles.safeArea}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
          <Text style={styles.eyebrow}>UCSC SILICON VALLEY CAMPUS</Text>
          <Text style={styles.title}>BELT Campus Guide</Text>
          <Text style={styles.subtitle}>
            Point the camera toward your surroundings, ask a question, and BELT will answer aloud.
          </Text>

          <View style={styles.cameraCard}>{cameraContent()}</View>

          <View style={styles.rowCard}>
            <View style={styles.rowText}>
              <Text style={styles.rowTitle}>Include camera image</Text>
              <Text style={styles.rowDescription}>Only one photo is sent when you ask.</Text>
            </View>
            <Switch value={includeCamera} onValueChange={setIncludeCamera} />
          </View>

          <Pressable
            style={[
              styles.talkButton,
              recorderState.isRecording && styles.stopButton,
              isSending && styles.disabledButton,
            ]}
            onPress={handleTalkButton}
            disabled={isSending}
          >
            {isSending ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.talkButtonText}>
                {recorderState.isRecording ? '■ Stop and ask BELT' : '● Start talking'}
              </Text>
            )}
          </Pressable>

          <Text style={styles.orText}>or type a question</Text>
          <TextInput
            style={styles.input}
            value={typedQuestion}
            onChangeText={setTypedQuestion}
            placeholder="Example: What room is this?"
            placeholderTextColor="#7c8797"
            multiline
            editable={!isSending && !recorderState.isRecording}
          />
          <Pressable
            style={[styles.askButton, isSending && styles.disabledButton]}
            onPress={handleTypedQuestion}
            disabled={isSending || recorderState.isRecording}
          >
            <Text style={styles.askButtonText}>Ask with text</Text>
          </Pressable>

          <View style={styles.statusCard}>
            <Text style={styles.statusLabel}>STATUS</Text>
            <Text style={styles.statusText}>{status}</Text>
          </View>

          {transcript ? (
            <View style={styles.messageCard}>
              <Text style={styles.messageLabel}>YOU SAID</Text>
              <Text style={styles.messageText}>{transcript}</Text>
            </View>
          ) : null}

          {answer ? (
            <View style={[styles.messageCard, styles.answerCard]}>
              <Text style={styles.messageLabel}>BELT</Text>
              <Text style={styles.messageText}>{answer}</Text>
              <Pressable style={styles.replayButton} onPress={() => speakAnswer(answer)}>
                <Text style={styles.replayButtonText}>Replay answer</Text>
              </Pressable>
            </View>
          ) : null}

          <Text style={styles.privacyText}>
            Privacy: the app sends a photo and/or recording only after you press Ask. Avoid recording
            people without their permission.
          </Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f4f7fb',
  },
  container: {
    padding: 20,
    paddingBottom: 48,
  },
  eyebrow: {
    marginTop: 8,
    color: '#315c8d',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  title: {
    marginTop: 4,
    color: '#132238',
    fontSize: 32,
    fontWeight: '800',
  },
  subtitle: {
    marginTop: 8,
    marginBottom: 18,
    color: '#526075',
    fontSize: 16,
    lineHeight: 23,
  },
  cameraCard: {
    height: 340,
    overflow: 'hidden',
    borderRadius: 22,
    backgroundColor: '#dfe7f1',
  },
  camera: {
    flex: 1,
  },
  cameraMessage: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  cameraMessageText: {
    marginBottom: 14,
    color: '#3d4b60',
    textAlign: 'center',
    fontSize: 15,
    lineHeight: 21,
  },
  rowCard: {
    marginTop: 14,
    padding: 16,
    borderRadius: 16,
    backgroundColor: '#ffffff',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  rowText: {
    flex: 1,
    paddingRight: 16,
  },
  rowTitle: {
    color: '#15243a',
    fontSize: 16,
    fontWeight: '700',
  },
  rowDescription: {
    marginTop: 3,
    color: '#68758a',
    fontSize: 13,
  },
  talkButton: {
    minHeight: 58,
    marginTop: 16,
    borderRadius: 18,
    backgroundColor: '#1769e0',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  stopButton: {
    backgroundColor: '#c53d43',
  },
  disabledButton: {
    opacity: 0.55,
  },
  talkButtonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
  },
  orText: {
    marginVertical: 13,
    color: '#788497',
    textAlign: 'center',
    fontSize: 13,
    fontWeight: '600',
  },
  input: {
    minHeight: 82,
    borderWidth: 1,
    borderColor: '#ced7e3',
    borderRadius: 16,
    backgroundColor: '#ffffff',
    color: '#16243a',
    padding: 14,
    fontSize: 16,
    textAlignVertical: 'top',
  },
  askButton: {
    minHeight: 48,
    marginTop: 10,
    borderRadius: 15,
    backgroundColor: '#243b5c',
    alignItems: 'center',
    justifyContent: 'center',
  },
  askButtonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '800',
  },
  smallButton: {
    borderRadius: 12,
    backgroundColor: '#1769e0',
    paddingHorizontal: 16,
    paddingVertical: 11,
  },
  smallButtonText: {
    color: '#ffffff',
    fontWeight: '800',
  },
  statusCard: {
    marginTop: 18,
    padding: 15,
    borderRadius: 15,
    backgroundColor: '#e9eef6',
  },
  statusLabel: {
    color: '#64738a',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
  },
  statusText: {
    marginTop: 5,
    color: '#263750',
    fontSize: 15,
    lineHeight: 21,
  },
  messageCard: {
    marginTop: 14,
    padding: 17,
    borderRadius: 17,
    backgroundColor: '#ffffff',
  },
  answerCard: {
    borderWidth: 1,
    borderColor: '#bad0ee',
    backgroundColor: '#eef5ff',
  },
  messageLabel: {
    color: '#315c8d',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1,
  },
  messageText: {
    marginTop: 7,
    color: '#17263d',
    fontSize: 16,
    lineHeight: 23,
  },
  replayButton: {
    alignSelf: 'flex-start',
    marginTop: 12,
    borderRadius: 12,
    backgroundColor: '#d6e6fb',
    paddingHorizontal: 13,
    paddingVertical: 9,
  },
  replayButtonText: {
    color: '#214c82',
    fontSize: 13,
    fontWeight: '800',
  },
  privacyText: {
    marginTop: 18,
    color: '#788497',
    fontSize: 12,
    lineHeight: 18,
    textAlign: 'center',
  },
});
