SELECT d.Patient, d.Date, d.mainDgCode, d.OtherDgCode
FROM `ehh2022.challenge6.Diagnose` d
WHERE d.mainDgCode LIKE "N18%" OR d.OtherDgCode LIKE "N18%"
