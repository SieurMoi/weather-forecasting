import os
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def station_choice():
    stations = []
    for root, dirs, files in os.walk("./data"):
        for file in files:
            if file.endswith(".xlsx"):
                stations.append(file)

    stationsNumber = len(stations)
    isStationChosen = False
    while (not isStationChosen):
        print("\nList of stations")
        for i in range(stationsNumber):
            print('\t' + str(i + 1) + ' - ' + stations[i])

        stationChosen = int(input("Select a station number : "))
        if (stationChosen in range(1, stationsNumber + 1)):
            isStationChosen = True

    print(stations[stationChosen - 1])
    return "./data/" + stations[stationChosen - 1]

def param_choice(parameters):
    isParamChosen = False
    
    while(not isParamChosen):
        print("\nList of parameters")
        for i in range(len(parameters)):
            print("\t" + str(i + 1) + ' - ' + parameters[i])
        paramChosen = int(input("Select watched parameter : "))
        if (paramChosen in range(1, len(parameters) + 1)):
            isParamChosen = True
    
    print(parameters[paramChosen - 1])
    return parameters[paramChosen - 1]

if __name__ == '__main__':
    parameters = [
        'MAX_TEMPERATURE_C', 'MIN_TEMPERATURE_C', 
        'WINDSPEED_MAX_KMH','TEMPERATURE_NOON_C',
        'PRECIP_TOTAL_DAY_MM','PRESSURE_MAX_MB']

    stationChosen = station_choice()
    paramChosen = param_choice(parameters)

    # PRETRAITEMENT DES DONNEES
    data = pd.read_excel(stationChosen, index_col='date', parse_dates=True)
    comparative = pd.read_excel("./data/export-biscarrosse2022_.xls", index_col='date', parse_dates=True)

    for i in data:
        if (not (data[i].name in parameters)):
            data.drop([i], axis=1, inplace=True)
    data['day'] = data.index.dayofyear

    startDate = '2022-01-01'
    newData = pd.DataFrame(data.groupby([data.day]).mean(), columns = parameters)
    newData['date'] = pd.date_range(startDate, periods=366, freq='D')
    newData.set_index('date', drop=True, inplace=True)
    data.drop(['day'], axis=1, inplace=True)

    data.index.freq = "D"
    data["target"] = data.shift(-1)[paramChosen]
    data = data.iloc[:-1,:].copy()
    newData["target"] = newData.shift(-1)[paramChosen]
    newData = newData.iloc[:-1,:].copy()

    comparative.index.fred = "D"
    comparative["target"] = comparative.shift(-1)[paramChosen]
    comparative = comparative.iloc[:-1, :].copy()


    trainingSize = len(data)
    print("Training size : " + str(trainingSize))
    testSize = len(newData) - 1
    print("Test size : " + str(testSize))

    X_train = data[parameters]
    Y_train = data["target"]
    X_test = newData[parameters]
    Y_test = newData["target"]
    X_comp = comparative[parameters]
    Y_comp = comparative["target"]

    C = 40

    model = SVR(C=C)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_pred = pd.Series(Y_pred, index=newData.index)
 
    graphicData = pd.concat([Y_test, Y_pred], axis=1)
    graphicData.columns = ["actual", "predictions"]
    print(model.score(X_test, Y_test))
    Y_pred.columns = ['predictions']
    graphicData.plot(figsize=(9,7))
    plt.title("Comparaison moyenne et prédiction sur un an")
    plt.xlabel("Date")
    plt.ylabel(paramChosen)
    plt.show()

    Y_pred.plot(figsize=(9,7))
    plt.title("Prédiction sur un an")
    plt.xlabel("Date")
    plt.ylabel(paramChosen)
    plt.show()

    compGraphic = pd.concat([Y_comp, Y_pred], axis=1)
    compGraphic.columns = ["real", "predictions"]
    compGraphic.plot(figsize=(9,7))
    plt.title("Comparaison réel et prédiction sur un an")
    plt.xlabel("Date")
    plt.ylabel(paramChosen)
    plt.show()


    


    


