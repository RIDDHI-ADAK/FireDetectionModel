#include <DHT.h>

// Define the pin where the DHT11 data pin is connected
#define DHTPIN 2     // DHT11 data pin connected to digital pin 2

// Define the type of sensor (DHT11 in this case)
#define DHTTYPE DHT11   


DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);

  // Initialize the DHT sensor
  dht.begin();

  // Setup pin for LED and analog input
  pinMode(13, OUTPUT); // LED pin
  pinMode(A0, INPUT);  // Analog sensor pin
}

void loop() {
  // Read the analog temperature (from A0)
  int temp = analogRead(A0);  // Read value from analog pin A0

  

  float humidity = dht.readHumidity();
  float temperatureC = dht.readTemperature();  
  float temperatureF = dht.readTemperature(true); 

  if (isnan(humidity) || isnan(temperatureC)) {
    Serial.println("Failed to read from DHT sensor!");
  } else {
  
    Serial.print(temperatureC);
    Serial.print(",");
    Serial.println(temp);
  }

  if (temperatureC > 40 && temp > 100) {
    digitalWrite(13, HIGH);  
  } else {
    digitalWrite(13, LOW);   
  }


  delay(2000);  
}
