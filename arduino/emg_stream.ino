void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Sensor Reading...");
}

void loop() {
  int sensor1 = analogRead(A0); // Lower lateral forearm
  int sensor2 = analogRead(A1); // Upper medial forearm

  Serial.print(sensor1);
  Serial.print(",");
  Serial.println(sensor2);

  delay(50); // 20 readings per second
}
