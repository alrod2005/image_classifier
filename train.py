import torch
from torchvision import transforms, datasets
import argparse
from util import generate_model, build_classifier

def load_and_transform_data(train_dir, valid_dir):
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    return train_dataloader, valid_dataloader, train_dataset.class_to_idx

def validate(model, valid_dataloader, criterion, device):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for i, (inputs, labels) in enumerate(valid_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def do_training(epochs, print_every, model, train_dataloader, valid_dataloader, optimizer, criterion, device):
    steps = 0
    model.to(device)
    
    for e in range(epochs):
        loss_so_far = 0
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_so_far += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate(model, valid_dataloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(loss_so_far/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_dataloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloader)))

                loss_so_far = 0

                model.train()
                
def save_checkpoint(arch, save_dir, input_size, output_size, hidden_units, dropout):
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_units': hidden_units,
                  'dropout': dropout,
                  'state_dict': model.classifier.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    
def main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")  
    
    # Define location of image data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define defaults
    input_size = 25088
    output_size = 102
    dropout = 0
    
    print_every = 40   
    
    train_dataloader, valid_dataloader, class_to_idx = load_and_transform_data(train_dir, valid_dir)
    
    model = generate_model(arch)
    if not model:
        print("Architecture not supported.")
        return
    
    model.class_to_idx = class_to_idx
    
    model.classifier = build_classifier(input_size, hidden_units, output_size, dropout)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    do_training(epochs, print_every, model, train_dataloader, valid_dataloader, optimizer, criterion, device)
    save_checkpoint(model, save_dir, input_size, output_size, hidden_units, dropout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an image classifier using pretrained architecture.',
    )
    
    parser.add_argument('data_directory', default='flowers')
    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--hidden_units', default=[])
    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--gpu', action='store_true', default=True)
    input_args = parser.parse_args()

    main(input_args.data_directory, input_args.save_dir, input_args.arch,
         input_args.learning_rate, input_args.hidden_units, input_args.epochs, input_args.gpu)