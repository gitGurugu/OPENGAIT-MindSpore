from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([str(package_dir)]):
    # Skip __pycache__ and other non-python files
    if module_name.startswith('_'):
        continue
    
    # import the module and iterate through its attributes
    try:
        module = import_module(f"{__name__}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if isclass(attribute):
                # Add the class to this package's variables
                globals()[attribute_name] = attribute
    except Exception as e:
        # Skip modules that can't be imported (may need further conversion)
        pass

