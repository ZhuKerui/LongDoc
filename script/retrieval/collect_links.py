from datasets import load_dataset

fineweb = load_dataset('HuggingFaceFW/fineweb', 'sample-10BT')

for record in fineweb['train']:
    print(record)
    break