# Imagine you're programming an agent that has these input and output modalities:

class Observation:
    vision: Image
    text: str
    energy: float
    
class Action:
    # -1 for left; 0 for still; +1 for right
    left_right_direction: int
    jump: bool

# Traditionally, you'd have to serialize these modalities by hand or hope your environment does for you

def step(obs: Observation) -> Action:
    
    # encode inputs
    image_enc = ViT(obs.vision)
    text_enc = gpt(obs.text)
    latent = concat([image_enc, text_enc, obs.energy])
    
    # actually make decision
    action = mlp(latent)
    
    # convert back to python objects
    left_right_direction = clip(action[0], -1, +1)
    jump = action[1] > 0
    
    return Action(left_right_direction, jump)


# TensorCode performs arbitrary object (modality) encoding and decoding

import tensorcode

def step(obs: Observation) -> Action:
    latent = tensorcode.encode(obs)
    action = mlp(latent)
    return tensorcode.decode(action, Action)
    
# Internally, tensorcode.encode and decode are implemented by recursively calling themselves down to fundamental Python types str, int, bool, list, set, dict. A perciever-style transformer uses the object's name as the query, its field names as the keys, and the associated objects themselves as values. Values that can be further reduced along tensor axes are directly attended to by an additional attention operation to minimize information loss 

# So an object like this turns into this:

class Observation:
    vision = Image(...)
    text = 'go to the door'
    energy = 0.98
    
query = 'Observation'
keys_and_values = [
    ('vision', Image(...)),
    ('text', 'go to the door'),
    ('energy', 0.98)
]

query = Tensor[768]
keys_and_values = [
    (Tensor[768], Tensor[256, 256, 3]),  # vision
    (Tensor[768], Tensor[4, 768]),  # text
    (Tensor[768], Tensor[1])  # energy
]

query = Tensor[12]
keys_and_values = [
    (Tensor[12], Tensor[256, 256, 3]),  # vision
    (Tensor[12], Tensor[4, 768]),  # text
    (Tensor[12], Tensor[1])  # energy
]

query = Tensor[12]
keys_and_values = [
    (Tensor[12], (Tensor[256, 12], Tensor[256, 16, 16, 3])),  # keys and values as image patches
    (Tensor[12], (Tensor[4, 12], Tensor[4, 768])),  # keys and values as tokens
    (Tensor[12], Tensor[1])  # no decomposition needed
]

# TensorCode simplifies integrating hand-engineered features, modifying existing modalities, and adding new ones

# example: observation

class Observation:
    vision: Image
    text: str
    energy: float

class Observation:
    vision: Image
    hand_engineered_signal_A: ClassA
    hand_engineered_signal_B: ClassB
    hand_engineered_signal_C: ClassC
    text: str
    energy: float

# example: action

class Action:
    # -1 for left; 0 for still; +1 for right
    left_right_direction: int
    jump: bool


DIRECTION = int
LEFT = -1
STILL = 0
RIGHT = +1

class Action:
    left_right_direction: DIRECTION
    jump: bool


class Action:
    direction2: tuple[DIRECTION, DIRECTION]
    jump: bool
    run: bool


# My future work on TensorCode will make it able to perform runtime code generation

# core operations: select, call, create, create_list, extend_list

x_dict = tc.create("a dictionary obtained from post-order traversal of X")
puppy = tc.select(query="a picture of a dog", animals)

# and differentiable programming

x = ... # some scalar tensor
y = f(x)

@tc.If(x == 1)
def MyIf:
  return x
@MyIf.Elif(x == 2)
def ElifClause():
  return x ** 2
@MyIf.Else(x == 3)
def ElseClause():
  return -x

dx = y.grad(x)

# Together, I hope these features will be instrumental in developing increasingly general cognitive architectures, that anyone can run on their personal computer, that any freshman can understand, and that any of us might use to develop artificial general intelligence

def step(self, obs: Observation) -> Action:

    # input and recurrent information
    obs_enc = tc.encode(obs)
    stm = tc.select(query=obs_enc@self.W_stm, values=self.short_term_memory)
    ltm = tc.select(query=obs_enc@self.W_stm, values=self.short_term_memory)
    
    # global workspace state
    perception_enc = self.perception_mlp(concat(obs_enc, stm, ltm))
    perception = tc.decode(perception_enc, Perception,
        context={'vision': obs.vision, 'prompt': obs.prompt})
        # `extras` kwarg useful when `perception_enc` bottleneck is too tight to squeze information through)
    
    # prediction
    prediction_enc = self.prediction_mlp(perception_enc)
    prediction = tc.decode(prediction_enc, Perception, context=perception)
    
    # get outputs
    self.monologue += perception.text
    self.short_term_memory.remember(tc.select('store short-term memories', prediction.thoughts))
    self.long_term_memory.remember(tc.select('store long-term memories', prediction.thoughts))
    action = prediction.action
    
    # feedback signals
    self.add_loss('reward', stopgrad(prediction.pain - prediction.pleasure)
    self.add_loss('pred_error', -tc.similarity(perception, self.prev_prediction)
    self.prev_prediction = prediction
    self.add_loss('pred_enc_error', -similarity(perception_enc, self.prev_prediction_enc)
    self.prev_prediction_enc = prediction_enc
    self.add_loss('stm_cognitive_load', self.l_stm_len*len(stm))
    self.add_loss('ltm_cognitive_load', self.l_ltm_len*len(ltm))
    
    return action
    
