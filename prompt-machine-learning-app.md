I would like to build a sophisticated web application for teaching the corporate principles of machine learning. I will use a simple regression to explore all of the key ideas. I am breaking it down into four ingredients as described below.

Must use Python streamlit for the code base

## Machine Learning in a Nutshell

Machine learning is about enabling computers to recognize patterns in data so they can make accurate predictions or decisions without being explicitly programmed for every scenario. At its core, it involves feeding a model real-world examples encoded in a dataset and allowing the computer to learn the relationship between inputs and outputs. Once trained, the model can apply that learned relationship to make predictions on new, unseen data.

To build a machine learning model, we need four key ingredients:

1.  **A mathematical model:** This defines the form of the mathematical function we’ll use to relate inputs to outputs, for example, a straight line (linear) or a more flexible structure like a neural network (non-linear).
2.  **Training data:** A collection of real-world examples that pair inputs with their corresponding outputs. The quality and relevance of this data are crucial to how well the model can learn and make accurate predictions.
3.  **A loss function:** A mathematical expression that measures how far off the model’s predictions are from the correct answers. It provides feedback to help the model improve over time.
4.  **A training algorithm:** A step-by-step procedure that combines the first three ingredients in a way that minimizes the prediction errors produced by the model. This is where the so-called *learning* takes place.

The app will consist of five interconnected panes as described below:
  -  0. User input pane where users can use sliders or hardcode in specific values for the weights the learning rate and other dials you think makes sense.
  -  1. show everthing related to ingredient one: mathematical function in this case, a simple linear regression 
    - show both the symbolic mathematical equation in latex, as well as an updated fitted model as the weights are adjusted
  -  2. a training data set with five data points, consisting of square footage and Home value pairs. Allow the user to edit the input data points.
  -  3. the MSE loss in both symbolic notation as well as a table, showing the calculations of the example by example loss as well as total MSE loss for all examples for a given set of weight
  -  4. gradient descent
    - Initial values for w0 and w1 determine your starting point on the surface
    - update rule in latex
    - update rule with actual caculations using MSE gradient
    - generic gradient notation followed by MSE specific gradient. 
    - show actual calculation given weights and training data
    
    
Tie all this together so that if values on any pane changes, everything flows through to all the other panes
