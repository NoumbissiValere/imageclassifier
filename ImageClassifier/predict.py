import utility
import torch
import numpy as np



def predict():
    args = utility.get_predict_inputs()
    image = utility.process_image(args.imagepath)
    
    model = utility.load_checkpoint(args.checkpoint)
    model.to(args.gpu)
    
    image.unsqueeze_(0)
    image = image.to(args.gpu)
    output = model.forward(image)
    ps = torch.exp(output)
    props, index = ps.topk(args.top_k)
    
    with torch.no_grad():
        props, index = props.to('cpu')[0].detach().numpy(), index.to('cpu')[0].detach().numpy()
    
    idx_to_class = {idx:class_name for class_name, idx in model.class_to_idx.items()}
    
    cats = []
    
    for i in index:
        cats.append(idx_to_class[i])
    
    if args.category_names:
        cat_to_name = utility.label_mapping(args.category_names)
        names = []

        for cat in cats:
            names.append(cat_to_name[cat])

        print(*props)
        print(*names)
    
    else:
        print(*props)
        print(*cats)
                                        
                                        
                                        
if __name__ == "__main__":
    
    predict()
    