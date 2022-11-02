import torch
import pickle 
import sys 
from argparse import ArgumentParser 
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_info()
transformers.utils.logging.set_verbosity_warning()

if __name__ == '__main__': 
    parser= ArgumentParser() 
    parser.add_argument('--type', default= 'small', type= str)
    parser.add_argument('--save-conv', default= True, type= bool)
    parser.add_argument('--device', default= 'cuda', type= str)
    try: 
        args= parser.parse_args() 
    except:
        parser.print_help()
        sys.exit(0)

    assert args.type in ['small', 'medium', 'large']

    device= torch.device(args.device)
    if device.type=='cuda': 
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    # Load model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{args.type}")
    model = AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{args.type}")
    model.to(device)
    model.eval() 

    step = 0
    convertation= dict() 
    while True:
        inputs = input(">> User: ")
        if inputs.startswith("stopconv".lower()):
            print("See ya!")
            break
        new_user_input_ids = tokenizer.encode(inputs + tokenizer.eos_token, return_tensors='pt').to(device)
        bot_input_ids = torch.cat([gen_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids  
        # gen_ids = model.generate(
        #     bot_input_ids.cuda(), 
        #     max_length=200, 
        #     pad_token_id=tokenizer.eos_token_id,
        #     ).cuda()  
        gen_ids = model.generate(
            bot_input_ids.to(device), 
            max_length=200, 
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.8,
            temperature = 0.8
            ).to(device) 
        step += 1
        print(">>Bot: {}".format(tokenizer.decode(gen_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        print()
