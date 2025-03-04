int vibratorPin = 7;  // Pin connected to the mini coin vibrator

void setup() {
  Serial.begin(9600);  // Start serial communication at 9600 baud
  pinMode(vibratorPin, OUTPUT);  // Set the vibrator pin as an output
  Serial.println("Setup complete. Waiting for commands...");
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();  // Read the incoming byte
    
    if (command == '1') {
      digitalWrite(vibratorPin, HIGH);  // Turn on the vibrator
      Serial.println("Vibrator ON");
    } else if (command == '0') {
      digitalWrite(vibratorPin, LOW);   // Turn off the vibrator
      Serial.println("Vibrator OFF");
    } else {
      Serial.println("Unknown command");
    }
  }
}
