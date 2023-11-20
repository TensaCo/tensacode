



x = f(y, z)

model.add_loss(l(x))


def foo(x,g,y,u):
  ...
  if(make_decision(g,y,z)):
    ...
  else:
    ...

  ...
  ...

  add_loss(...)


def act(x, y, energy, holding_box):
  def should_move_left(x, y, holding_box):
    ...
    return True # or False

  if tensorcode.make_decision(x, y, holding_box):
    ...
  else:
    ...


  if goal:
    tensorcode.add_loss(-1)


def make_decision(*args):
  my_tensor = tensorcode.encode(*args)
  decision = classifier(my_tensor, bool)

def select(query, *options):
  encoded_options = [tensorcode.encode(option) for option in options]
  scores = [score_fn(option) for option in encoded_options]
  return max((score, ))


dog_picture = tensorcode.select("the largest dog", image1, image2, image3)

answer = tensorcode.select(question, *options)

def function(*args: Types...) -> Type:
  ...

# programming 2.0
def function(arg1: "a simple representation", full: "a complex representation")




my_obj:
  __dict__:
    a: Object() = "235"
    gr: 4343 = 1
__subclass__:


obj = Object()
obj.__setattr__('foo', 1)
obj.__dict__ == {'foo': 1}




ast(obj) == {}


Graph<Verts, Edges>
Verts := obj
Edges := Tuple[str, obj]