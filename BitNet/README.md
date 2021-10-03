
1. Kör BitBot c++ för att ladda ner och skapa både simulator- och träningsdata.
2. Träna med learn_sections.ipynb i jupyter lab.
3. Kör **make_predictions.py**
  - in: *training_data.pickle*
  - out: *predictions.pickle*
4. Kör simulator.py

Old:

1. **download.py**
   - out: *klines/{symbol}.hdf*
   
1. **filter_by_volume.py**
   - in: *klines/{symbol}.hdf*
   - out: *filtered_symbols.pickle*

1. **make_intrinsic_events.py**
   - in: *filtered_symbols.pickle*
   - in: *klines/{symbol}.hdf*
   - out: *intrinsic_events.pickle*
   
1. **make_indicators.py** - set end_timestamp
   - in: *intrinsic_events.pickle*
   - out: *indicators/{symbol}.pickle*
   
1. **make_training_data.py**
   - in: *indicators/{symbol}.pickle*
   - out: *training_data.pickle*

1. **make_predictions.py**
   - in: *training_data.pickle*
   - out: *predictions.pickle*
   
Note:
python-binance before v1.0.12 doesn't have support for python 3.9
