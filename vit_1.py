# %% [markdown]
# #### Vision Transformer from Scratch

# %%
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as dataloader
import torch.nn as nn

# %%
# preprocessing step that converts images into PyTorch tensors, usually scaling pixel values to a numeric tensor format
transform_op = transforms.Compose([transforms.ToTensor()])

# %%
train_ds = torchvision.datasets.MNIST(root='/DATA/pranta_2411ai09/gemma', train=True,download=True,transform=transform_op)
val_ds = torchvision.datasets.MNIST(root='/DATA/pranta_2411ai09/gemma', train=False,download=False,transform=transform_op)

# %%
#define variables
num_classes = 10
batch_size = 16
num_channels = 1 # since MNIST dataset is black & white. For color, it is 3(RGB)
img_size = 28 #Changes on dataset that we use. For MNIST it is 28x28
patch_size = 7 #7x7
num_patches = (img_size//patch_size)**2 #(28/7)*(28/7) = 16
lr = 0.001
# each patch will be converted into one vector
emdedding_dim = 64
num_heads = 4
num_layer = 4
num_mlp_heads = 128 # 64*2
num_epoch = 5

# %%
# divide the data into batches
train_dl = dataloader.DataLoader(train_ds,batch_size=batch_size,shuffle=True)
val_dl = dataloader.DataLoader(val_ds,batch_size=batch_size,shuffle=True)

# %% [markdown]
# #### part 1 patch embedding
# #### part 2 Transformer Encoder
# #### part 3 MLP Heads

# %%
# part 1 patch embedding
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedd = nn.Conv2d(num_channels,emdedding_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        #patch emdedding
        x = self.patch_embedd(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x

# %%
data_point,label = next(iter(train_dl))

# %%
data_point.shape

# %%
patch_embedd = nn.Conv2d(num_channels,emdedding_dim,kernel_size=patch_size,stride=patch_size)
print(patch_embedd(data_point).shape)
patch_embedd_flatten = patch_embedd(data_point).flatten(2)
print(patch_embedd_flatten.shape)

# %% [markdown]
# ### Patch Embedding for MNIST ViT
# 
# We start with an MNIST image batch of shape:
# 
# - `B x 1 x 28 x 28`
# 
# where:
# 
# - `B` = batch size
# - `1` = number of input channels
# - `28 x 28` = image height and width
# 
# ---
# 
# ### Step 1: Apply `Conv2D`
# 
# We use a convolution layer with:
# 
# - `kernel_size = 7`
# - `stride = 7`
# - `embedding_dim = D`
# 
# This splits each image into non-overlapping `7 x 7` patches.
# 
# Since:
# 
# - `28 / 7 = 4`
# 
# the image is divided into:
# 
# - `4 x 4 = 16` patches
# 
# So after `Conv2D`, the output shape becomes:
# 
# - `B x D x 4 x 4`
# 
# ---
# 
# ### Step 2: Apply `flatten(2)`
# 
# Now we flatten the last two dimensions:
# 
# - `4 x 4 = 16`
# 
# So the shape becomes:
# 
# - `B x D x 16`
# 
# ---
# 
# ### Step 3: Apply `transpose(1, 2)`
# 
# We swap the last two dimensions so that patches come before embedding dimension.
# 
# Final shape becomes:
# 
# - `B x 16 x D`
# 
# ---
# 
# ### Final Output
# 
# So the patch embedding block converts:
# 
# - `B x 1 x 28 x 28`
# 
# into:
# 
# - `B x 16 x D`
# 
# This means:
# 
# - each image becomes a sequence of `16` patch tokens
# - each token has embedding dimension `D`
# 
# ---
# 
# ### Small Shape Diagram
# 
# ```text
# Input image batch
# B x 1 x 28 x 28
#         |
#         |  Conv2D (kernel_size=patch_size, stride=patch_size)
#         v
# B x D x 4 x 4
#         |
#         |  flatten(2)
#         v
# B x D x 16
#         |
#         |  transpose(1, 2)
#         v
# B x 16 x D
# ```

# %%
# Part 2
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(emdedding_dim)
        self.layer_norm2 = nn.LayerNorm(emdedding_dim)
        self.mha = nn.MultiheadAttention(emdedding_dim,num_heads,batch_first = True)
        self.mlp = nn.Sequential(
            nn.Linear(emdedding_dim,num_mlp_heads),
            nn.GELU(),
            nn.Linear(num_mlp_heads,emdedding_dim),
        )
    def forward(self,x):
        residual1 = x
        x = self.layer_norm1(x)
        x = self.mha(x,x,x)[0]
        x = x + residual1
        
        residual2 = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual2
        return x
        

# %%
# part 3
class MLP_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emdedding_dim)
        self.mlp_head = nn.Linear(emdedding_dim,num_classes)
    def forward(self,x):
        x = self.layer_norm(x)
        x = self.mlp_head(x)
        return x

# %% [markdown]
# ### CLS Token
# - The initial cls_token shape is 1 x 1 x D.
# - It is expanded across the batch so each image gets one cls token.
# - If your image has 16 patches, the sequence becomes 16 + 1 = 17 tokens.
# - So the final token sequence per batch is B x 17 x D.
# - So the number of tokens increases by 1, not the number of patches.

# %%
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_emb = PatchEmbedding()
        # Shape of CLS token: 1 x 1 x D. After expanding for a batch: B x 1 x D. It is placed at the front of the patch sequence. volkswagen
        self.cls_token = nn.Parameter(torch.randn(1,1,emdedding_dim)) 
        self.pos_emb = nn.Parameter(torch.randn(1,1+num_patches,emdedding_dim))
        self.transformer_blocks = nn.Sequential(*[TransformerEncoder() for _ in range(num_layer)])
        self.mlp_head = MLP_head()
    def forward(self,x):
        x = self.patch_emb(x)
        B = x.size(0)
        class_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((class_tokens,x),dim=1)
        x = x + self.pos_emb
        x = self.transformer_blocks(x)
        x = x[:,0]
        x = self.mlp_head(x)
        return x
        

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
device

# %%
model = VisionTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

# %%
for epoch in range(num_epoch):
    model.train()
    total_loss = 0
    correct_epoch = 0
    total_epoch = 0

    print(f"\nEpoch {epoch + 1}")

    for batch_idx, (images, labels) in enumerate(train_dl):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()

        correct_epoch += correct
        total_epoch += labels.size(0)

    train_acc = 100.0 * correct_epoch / total_epoch
    avg_loss = total_loss / len(train_dl)

    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

# %%
model.eval()
val_correct = 0
val_total = 0

with torch.no_grad():
    for images, labels in val_dl:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

val_acc = 100.0 * val_correct / val_total
print(f"Validation Accuracy: {val_acc:.2f}%")
#CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 vit_1.py


