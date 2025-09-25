DATASET=simulated
SID=idx

general:
	python main.py -t -g -o outputs/"${DATASET}" -d ../dataset/"${DATASET}" -s ../dataset/"${DATASET}.csv" -i ${SID}

simulated:
	python ../dataset/simulator.py -l 5000 -n 2500 -o simulated -m 5000 -d
	python main.py -t -g -o outputs/simulated -d ../dataset/simulated_split/train -s ../dataset/simulated_split/train.csv -i idx -c config/simulated.yaml -n 500

baseball:
	python ../dataset/process_baseball.py 
	python main.py -t -o outputs/baseball -d ../dataset/baseball/encoded_split/train -s ./dataset/baseball/encoded_split/train.csv -i players_id -c config/baseball.yaml
	python main.py -g -o outputs/baseball -c config/baseball.yaml -n 826

weather:
	python ../dataset/process_weather.py
	python main.py -t -g -o outputs/weather -d ../dataset/weather_data/train -c config/weather.yaml -n 30

sensors:
	python ../dataset/process_sensors.py
	python main.py -t -g -o outputs/sensors -d ../dataset/sensors_data/train -c config/sensors.yaml -n 5551

stocks:
	python ../dataset/process_stocks.py
	python main.py -t -g -o outputs/stocks -d ../dataset/stocks_data/train -c config/stocks.yaml -n 800