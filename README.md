# Self attention based chess neural network
<img width="1550" height="733" alt="image" src="https://github.com/user-attachments/assets/45407fa7-f0a6-4b8e-ba84-a30b93923985" />

## Input preparation
<img width="548" height="111" alt="image" src="https://github.com/user-attachments/assets/31797e7b-0bc7-42a9-898e-e7a38dbfa106" />
Take the raw board, a matrix with values between 0-5 with negative values signifying a black piece, and encode it sparsely. The sparse encoding is projected to the hidden dimension size to facilitate matrix multiplication. Encodings based on sine and cosine waves are then added to denote position as this information can be lost during the self attention step.

## Attention
<img width="424" height="127" alt="image" src="https://github.com/user-attachments/assets/74cd5b06-24e5-40b1-8658-16564689b9de" />

Attention is defined by the equation
#### $$Attention(Q, K, V) = softmax(\frac{(Q \cdot K^T)}{\sqrt{d_k})})V $$
Where $Q$, $K$, and $V$ are all the current board state after a forward pass through three different fully connected neural networks. Division by $d_k$ scales down the result to prevent vanishing gradients. Self attention computes the relationship between every single square on the board. It is followed by a dropout layer to help the model disregard unimportant connections on the board. These outputs are then normalized to have a mean of 0 and a variance of around 1.

## Fully Connected Layers
<img width="549" height="115" alt="image" src="https://github.com/user-attachments/assets/e903645a-71ad-4fff-ad8d-22b52548ab8d" />

Five seperate fully connected trainable layers allow the model to learn complex positions.

## Final Post Processing
<img width="406" height="108" alt="image" src="https://github.com/user-attachments/assets/58d523f8-07aa-4d1a-a67c-41bda8330ddb" />

Scale the vector to a scalar with one final linear layer, then apply a tanh function to the output to keep values between -1 and 0.
