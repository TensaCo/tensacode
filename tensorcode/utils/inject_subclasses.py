from typing import Callable, Dict, List, Optional, Tuple

INJECTED: Dict[Tuple[type, str], List[type]] = dict()

def inject_epilog_into_subclasses( owner: type, method: str, fn: Callable):
  inject_subclasses(owner=owner,method=method,post=fn)

def inject_prolog_into_subclass( owner: type, method: str, fn: Callable):
  inject_subclasses(owner=owner,method=method,post=fn)

def inject_subclasses(
  owner: type,
  method: str,
  pre: Optional[Callable],
  post: Optional[Callable]):
  """Attatches `injection` onto the end of the init method 
  of all subclasses of `owner`"""

  if (owner, method) not in INJECTED:
    INJECTED[(owner, method)] = list()

  for subclass in owner.__subclasses__:
    if subclass not in INJECTED[(owner, method)]:

      old = getattr(subclass, method)
      def new(*args, **kwargs):
        pre(*args, **kwargs)
        old(*args, **kwargs)
        post(*args, **kwargs)
      setattr(subclass, method, new)

      INJECTED[owner].append(subclass)