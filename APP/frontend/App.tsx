import { CameraView, useCameraPermissions } from "expo-camera";
import * as Speech from "expo-speech";
import { useEffect, useRef, useState } from "react";
import {
  Animated,
  Image,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

export default function Index() {
  const [permission, requestPermission] = useCameraPermissions();
  const [message, setMessage] = useState(
    "Hi! I'm BELT. Tap me to hear me talk!"
  );
  const [question, setQuestion] = useState("");
  const floatingAnimation = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(floatingAnimation, {
          toValue: -12,
          duration: 900,
          useNativeDriver: true,
        }),
        Animated.timing(floatingAnimation, {
          toValue: 0,
          duration: 900,
          useNativeDriver: true,
        }),
      ])
    );

    animation.start();

    return () => animation.stop();
  }, [floatingAnimation]);

  function makeBeltTalk() {
  const response = question.trim()
    ? `You asked: ${question}. I will be connected to the AI backend next.`
    : "Hi! I am BELT, your UCSC campus guide. Ask me where you would like to go.";

  setMessage(response);

  Speech.stop();
  Speech.speak(response, {
    rate: 0.95,
    pitch: 1.05,
    language: "en-US",
  });
}

  if (!permission) {
    return (
      <View style={styles.loadingScreen}>
        <Text style={styles.loadingText}>Loading BELT...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionScreen}>
        <Text style={styles.title}>BELT needs camera access</Text>

        <Text style={styles.permissionText}>
          The camera lets BELT see signs and surroundings.
        </Text>

        <Pressable style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.buttonText}>Allow camera</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView style={StyleSheet.absoluteFill} facing="back" />

      <SafeAreaView style={styles.overlay}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>BELT Campus Guide</Text>
          <Text style={styles.headerSubtitle}>Camera assistant</Text>
        </View>

        <View style={styles.robotArea}>
          <View style={styles.speechBubble}>
            <Text style={styles.speechText}>{message}</Text>
          </View>

          <Animated.View
            style={{
              transform: [{ translateY: floatingAnimation }],
            }}
          >
            <Pressable
              onPress={makeBeltTalk}
              style={({ pressed }) => [
                styles.robotButton,
                pressed && styles.robotPressed,
              ]}
            >
              <Image
  source={require("./assets/belt-robot.png")}
  style={styles.robotImage}
/>
            </Pressable>
          </Animated.View>
        </View>

        <View style={styles.bottomPanel}>
          <Text style={styles.instructions}>
            Tap BELT to hear a test response.
          </Text>
          
            <TextInput
              value={question}
              onChangeText={setQuestion}
              placeholder="Ask BELT a question..."
              placeholderTextColor="#9ca3af"
              style={styles.input}
            />

          <Pressable style={styles.talkButton} onPress={makeBeltTalk}>
            <Text style={styles.microphone}>🎤</Text>
            <Text style={styles.talkButtonText}>Talk to BELT</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "black",
  },

  overlay: {
    flex: 1,
    justifyContent: "space-between",
  },

  robotImage: {
    width: 85,
    height: 85,
    resizeMode: "contain",
  },

  input: {
    backgroundColor: "white",
    color: "#111827",
    paddingHorizontal: 16,
    paddingVertical: 13,
    borderRadius: 16,
    fontSize: 16,
    marginBottom: 12,
  },

  loadingScreen: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#101827",
  },

  loadingText: {
    color: "white",
    fontSize: 20,
  },

  permissionScreen: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 30,
    backgroundColor: "#101827",
  },

  title: {
    color: "white",
    fontSize: 25,
    fontWeight: "bold",
    textAlign: "center",
  },

  permissionText: {
    color: "#d1d5db",
    fontSize: 16,
    textAlign: "center",
    marginTop: 12,
    marginBottom: 25,
  },

  permissionButton: {
    backgroundColor: "#2563eb",
    paddingHorizontal: 25,
    paddingVertical: 14,
    borderRadius: 16,
  },

  buttonText: {
    color: "white",
    fontSize: 17,
    fontWeight: "bold",
  },

  header: {
    marginHorizontal: 18,
    marginTop: 10,
    padding: 14,
    borderRadius: 18,
    backgroundColor: "rgba(0, 0, 0, 0.55)",
  },

  headerTitle: {
    color: "white",
    fontSize: 22,
    fontWeight: "bold",
  },

  headerSubtitle: {
    color: "#d1d5db",
    fontSize: 14,
    marginTop: 2,
  },

  robotArea: {
    position: "absolute",
    right: 18,
    bottom: 225,
    alignItems: "flex-end",
  },

  speechBubble: {
    width: 240,
    backgroundColor: "white",
    padding: 14,
    borderRadius: 18,
    marginBottom: 14,
  },

  speechText: {
    color: "#111827",
    fontSize: 15,
    lineHeight: 20,
  },

  robotButton: {
    width: 115,
    height: 115,
    borderRadius: 58,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, 0.92)",
    borderWidth: 4,
    borderColor: "#60a5fa",
  },

  robotPressed: {
    transform: [{ scale: 0.94 }],
  },


  bottomPanel: {
    margin: 18,
    padding: 16,
    borderRadius: 22,
    backgroundColor: "rgba(0, 0, 0, 0.65)",
  },

  

  instructions: {
    color: "white",
    fontSize: 14,
    textAlign: "center",
    marginBottom: 12,
  },

  talkButton: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#2563eb",
    paddingVertical: 15,
    borderRadius: 18,
  },

  microphone: {
    fontSize: 22,
    marginRight: 9,
  },

  talkButtonText: {
    color: "white",
    fontSize: 17,
    fontWeight: "bold",
  },
});