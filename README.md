# TEST GENETIC CNN WITH BASE MODEL

Base on code <a href="https://github.com/H999/GeneticCNN-torch">Genetic CNN by TORCH</a> and add the def update_stages_state in Stages to change curent CNN state architecture of Individual

```python
class Stages(torch.nn.Module):
...
    def update_stages_state(self, new_gen):
        if type(new_gen) != tuple:
            raise ValueError("new gen must be tuple")
        for i in range(len(new_gen)):
            self.stages[i].binary_code = new_gen[i]
            self.stages[i].inputs, self.stages[i].outputs, self.stages[i].separated_connections = Stage.get_nodes_connections(self.num_stages[i], new_gen[i])
        self.gen = self.get_gen()
        self.gen_model = self.get_gen('model')
...
```

<img src="./img/drawio plot code format.drawio.png" title="class format flow"/>

## Idea

Change architecture of connection to reduce time of Genetic Algorithm, it only train the necessary architecture if the different of base model to approximately model not large

## Experiment

file log.csv and model.pt is log about training model {'S_1': '1-11', 'S_2': '1-11-111-1111'} and change architecture to some different architecture

## Conclude

- The different of base model to approximately model mean the connection between nodes in Stage and the contribution of the node if exist or not
- Each Stage have effect to the final model follow Conditional probability
- In some case the weight of the base model is suitable to approximately model more than base model (acc better, loss lesser)

## Test

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/H999/GeneticCNN-torch-test-with-base-model/blob/main/test.ipynb)

If want to test the new model, delete file log.csv and model.pt

# ENVIRONMENT REQUIREMENTS

- pytorch >= 1.9

