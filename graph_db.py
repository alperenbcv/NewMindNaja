import pandas as pd
from datetime import date, timedelta
import random

persons = []
names = ["Ali Yılmaz", "Ayşe Demir", "Mehmet Kaya", "Zeynep Şahin", "Ahmet Kurt",
         "Elif Koç", "Mustafa Aydın", "Fatma Yüce", "Burak Can", "Derya Öz"]
genders = ["Male", "Female"]
educations = ["Primary", "High School", "Bachelor", "Master", "PhD"]

for i in range(1, 11):
    persons.append({
        "personId": i,
        "name": names[i - 1],
        "age": random.randint(18, 70),
        "gender": genders[(i % 2)-1],
        "education": random.choice(educations)
    })

crimes = []
crime_types = ["Intentional Homicide", "Theft", "Robbery", "Fraud", "Drug Trafficking",
               "Bribery", "Kidnapping", "Sexual Assault", "Cybercrime", "Arson"]
articles = ["TCK 82", "TCK 141", "TCK 148", "TCK 157", "TCK 188",
            "TCK 252", "TCK 109", "TCK 102", "TCK 243", "TCK 170"]

for i in range(1, 11):
    crimes.append({
        "crimeId": 1000 + i,
        "type": crime_types[i - 1],
        "article": articles[i - 1],
        "date": (date(2023, 1, 1) + timedelta(days=random.randint(0, 365))).isoformat()
    })

committed = []
for i in range(1, 11):
    committed.append({
        "personId": i,
        "crimeId": 1000 + i
    })

df_persons = pd.DataFrame(persons)
df_crimes = pd.DataFrame(crimes)
df_committed = pd.DataFrame(committed)

df_persons.to_csv("persons.csv", index=False)
df_crimes.to_csv("crimes.csv", index=False)
df_committed.to_csv("committed.csv", index=False)
