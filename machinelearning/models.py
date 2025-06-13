from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim

class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        self.w = Parameter(ones(1, dimensions))
        

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
         
        return tensordot(x, self.w, dims=([1], [1]))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
         
        if self.run(x).item() >= 0:
            return 1
        return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            def has_converged():
                for sample in dataloader:
                    x = sample['x']
                    y = sample['label'].item()
                    prediction = self.get_prediction(x)
                    if prediction != y:
                        self.w.data += y * x.squeeze(0)
                        return False
                return True

            while not has_converged():
                pass


"""   2"""

class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
         
        super().__init__()
        config = {
            'batch_size': 128,
            'hidden_size': 128,
            'it_limit': 1200,
            'lr': 0.002,
            'layers_number': 3
        }
        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.it_limit = config['it_limit']
        self.lr = config['lr']
        self.layers_number = config['layers_number']
        self.input_layer = Linear(1, self.hidden_size)
        self.mid_layer = Linear(self.hidden_size, self.hidden_size)
        self.output_layer = Linear(self.hidden_size, 1)
        self.relu = relu


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
         

        x = self.relu(self.input_layer(x))
        for _ in range(self.layers_number):
            x = self.relu(self.mid_layer(x))
        x = self.relu(self.mid_layer(x))
        return self.output_layer(x)


    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
         

        loss = mse_loss(self.forward(x), y)
        return loss
        

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
         


        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = mse_loss  

        for epoch in range(self.it_limit):
            running_loss = 0.0
            for sample in data_loader:
                x, y = sample['x'], sample['label']

                optimizer.zero_grad()
                predictions = self(x)
                loss = loss_fn(predictions, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(data_loader)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

            if avg_loss < 0.01:
                print("Stopping early: Loss < 0.01")
                break



"""   3"""

class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10

        config = {
            'batch_size': 128,
            'hidden_size': 128,
            'it_limit': 1200,
            'lr': 0.002,
            'layers_number': 2
        }
         
        
        self.batch_size = config['batch_size']
        self.hidden_size = config['hidden_size']
        self.it_limit = config['it_limit']
        self.lr = config['lr']
        self.layers_number = config['layers_number']

        self.input_layer = Linear(input_size, self.hidden_size)
        self.mid_layer = Linear(self.hidden_size, self.hidden_size)
        self.output_layer = Linear(self.hidden_size, output_size)
        self.relu = relu


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        x = self.relu(self.input_layer(x))
        for _ in range(self.layers_number):
            x = self.relu(self.mid_layer(x))
        return self.output_layer(x)
 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        logits = self.run(x)
        targets = y.argmax(dim=1)
        return cross_entropy(logits, targets)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        target_val_acc = 0.98
        for epoch in range(self.it_limit):
            running_loss = 0.0
            for batch in data_loader:
                inputs, targets = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(inputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            val_acc = dataset.get_validation_accuracy()
            print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")
            if val_acc >= target_val_acc:
                print("Stopping early, target validation accuracy reached.")
                break


"""   4"""

class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
         

        self.batch_size = 128
        self.hidden_size = 254
        output_size = len(self.languages)
        self.lr = 0.0015

        self.input_layer = Linear(self.num_chars, self.hidden_size)
        self.mid_layer = Linear(self.hidden_size, self.hidden_size)
        self.output_layer = Linear(self.hidden_size, output_size)
        self.relu = relu


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
         


        for i in range(len(xs)):
            if i == 0:
                x = self.input_layer(xs[i])
                x = self.relu(x)
            else:
                s = self.input_layer(xs[i]) + self.mid_layer(x)
                x = self.relu(s)
        return self.output_layer(x)

    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
         

        args = torch.argmax(y, dim=1)
        logits = self.run(xs)
        loss_fn = cross_entropy
        return loss_fn(logits, args)

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
         
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        target_val_acc = 0.815

        for it in range(1000):
            total = 0
            for batch in data_loader:
                x = batch['x']
                x = movedim(x, 0, 1) 

                y = batch['label']   

                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                total += loss.item()
            
            val_acc = dataset.get_validation_accuracy()
            print(f"Epoch {it}, Val Acc = {val_acc:.4f}" )

            if val_acc >= target_val_acc:
                print("accuracy reached")
                break
        
"""   5"""
def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())

    H, W = input_tensor_dimensions
    kh, kw = weight_dimensions

    out_H = H - kh + 1
    out_W = W - kw + 1

    # Initialiser la sortie manuellement
    Output_Tensor = tensor([[0.0 for _ in range(out_W)] for _ in range(out_H)])

    for i in range(out_H):
        for j in range(out_W):
            patch = input[i:i+kh, j:j+kw]
            Output_Tensor[i, j] = (patch * weight).sum()

    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """

        input_size = 26 * 26

        self.hidden_size = 128
        self.batch_size = 128
        self.lr = 0.0001
        self.it_limit = 1000
        self.layers_number = 2

        self.input_layer = Linear(input_size, self.hidden_size)
        self.mid_layer = Linear(self.hidden_size, self.hidden_size)
        self.output_layer = Linear(self.hidden_size, output_size)
        self.relu = relu


    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous   s.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        x = self.relu(self.input_layer(x))
        for _ in range(self.layers_number):
            x = self.relu(self.mid_layer(x))
        return self.output_layer(x)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """

        predictions = self.run(x)
        labels = torch.argmax(y, dim=1)
        return cross_entropy(predictions, labels)
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        target_val_acc = 0.975

        for it in range(self.it_limit):
            total_loss = 0
            for batch in data_loader:
                x = batch['x']
                y = batch['label']

                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            

            val_acc = dataset.get_validation_accuracy()
            print(f"Epoch {it}, Val Acc = {val_acc:.4f}")

            if val_acc >= target_val_acc:
                print("Stopping early, target validation accuracy reached.")
                break

import torch
from torch.nn import Module, Linear
import torch.nn.functional as F


        
       


