
EHmax = 1000

def get_harvested_energy(time):
    hour =  int(time / 60) % 24
    if hour <= 7:
        return 0
    elif hour <= 10:
        return EHmax * (hour - 7) / 3
    elif hour <= 15:
        return EHmax * (1.2 - 0.2 * hour / 10)
    elif hour <= 18:
        return EHmax * (18 - hour) / 3
    else:
        return 0
    
sum = 0    
for i in range(25):
    en = get_harvested_energy(i * 60)
    sum += en

print(sum / EHmax)