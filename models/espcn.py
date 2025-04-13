class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, (upscale_factor ** 2) * 3, kernel_size=3, padding=1)  # Handle 3-channel output
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

def espcn_upscale(image):
    image = image/255
    image = torch.from_numpy(image).float().unsqueeze(dim=0).permute(0,3,1,2)

    model = ESPCN()
    checkpoint = torch.load("weights/model_espcn.pth", map_location=torch.device('cpu'))  # or 'cuda' if using GPU
    state_dict = checkpoint['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." from each key
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    with torch.no_grad():
        output = model(image)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255.0).clip(0, 255).astype("uint8")
    return output 