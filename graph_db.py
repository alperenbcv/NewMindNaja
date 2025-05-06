import pandas as pd
from datetime import date, timedelta
import random

names = ["Ali Yılmaz", "Ayşe Demir", "Mehmet Kaya", "Zeynep Şahin", "Ahmet Kurt",
         "Elif Koç", "Mustafa Aydın", "Fatma Yüce", "Burak Can", "Derya Öz"]
genders = ["Male", "Female"]
educations = ["Primary", "High School", "Bachelor", "Master", "PhD"]

persons = [{
    "personId": i,
    "name": names[i - 1],
    "age": random.randint(18, 70),
    "gender": genders[(i % 2) - 1],
    "education": random.choice(educations)
} for i in range(1, 11)]

crime_types = ["Intentional Homicide", "Theft", "Robbery", "Fraud", "Drug Trafficking",
               "Bribery", "Kidnapping", "Sexual Assault", "Cybercrime", "Arson"]
articles = ["TCK 82", "TCK 141", "TCK 148", "TCK 157", "TCK 188",
            "TCK 252", "TCK 109", "TCK 102", "TCK 243", "TCK 170"]

crimes = [{
    "crimeId": 1000 + i,
    "type": crime_types[i - 1],
    "article": articles[i - 1],
    "date": (date(2023, 1, 1) + timedelta(days=random.randint(0, 365))).isoformat()
} for i in range(1, 11)]

committed = [{
    "personId": i,
    "crimeId": 1000 + i
} for i in range(1, 11)]

court_names = [f"Criminal Court No.{i}" for i in range(1, 6)]
courts = [{
    "courtId": i,
    "name": court_names[i % len(court_names)]
} for i in range(1, 11)]

sentenced = [{
    "personId": i,
    "courtId": i,
    "sentenceYears": random.choice([3, 5, 8, 10, 12, 15, 18, 20, 25, 30])
} for i in range(1, 11)]

df_persons = pd.DataFrame(persons)
df_crimes = pd.DataFrame(crimes)
df_committed = pd.DataFrame(committed)
df_courts = pd.DataFrame(courts)
df_sentenced = pd.DataFrame(sentenced)

df_persons.to_csv("persons.csv", index=False)
df_crimes.to_csv("crimes.csv", index=False)
df_committed.to_csv("committed.csv", index=False)
df_courts.to_csv("courts.csv", index=False)
df_sentenced.to_csv("sentenced.csv", index=False)
