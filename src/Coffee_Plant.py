from Plant import Plant


class CoffeePlant(Plant):

    def __init__(self, ph: float, soil_temperature: float, soil_moisture: float, illuminance: float,
                 env_temperature: float, env_humidity: float, label: bool):
        self.ph = ph
        self.soil_temperature = soil_temperature
        self.soil_moisture = soil_moisture
        self.illuminance = illuminance
        self.env_temperature = env_temperature
        self.env_humidity = env_humidity
        self.label = label

    def to_string(self) -> str:
        return "Coffee Plant{" + \
            "Ph: " + self.ph + \
            "Soil{" + \
            "Temperature: " + self.soil_temperature + \
            ", Moisture: " + self.soil_moisture + \
            "}" + \
            ", Illuminance: " + self.illuminance + \
            ", Environment{" + \
            "Temperature: " + self.env_temperature + \
            ", Humidity: " + self.env_humidity + \
            "}" + \
            ", Label: " + "yes" if self.label else "no" + \
            "}"
