'''
## Nightly source code : https://github.com/keras-team/keras/blob/master/keras/applications/regnet.py#L909-L934
## Description : This implementation is the simplified version of RegNet.
## tag : 2022/02/17, JosefHuang (refactor)
'''
from arch_elements import RegNet
from metadata import MODEL_CONFIGS 


## I. RegNetX interfaces : 
#  from light-weight to heavy model
def RegNetX002(model_name="regnetx002",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x002"]["depths"],
        MODEL_CONFIGS["x002"]["widths"],
        MODEL_CONFIGS["x002"]["group_width"],
        MODEL_CONFIGS["x002"]["block_type"],
        MODEL_CONFIGS["x002"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX004(model_name="regnetx004",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x004"]["depths"],
        MODEL_CONFIGS["x004"]["widths"],
        MODEL_CONFIGS["x004"]["group_width"],
        MODEL_CONFIGS["x004"]["block_type"],
        MODEL_CONFIGS["x004"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX006(model_name="regnetx006",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x006"]["depths"],
        MODEL_CONFIGS["x006"]["widths"],
        MODEL_CONFIGS["x006"]["group_width"],
        MODEL_CONFIGS["x006"]["block_type"],
        MODEL_CONFIGS["x006"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX008(model_name="regnetx008",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x008"]["depths"],
        MODEL_CONFIGS["x008"]["widths"],
        MODEL_CONFIGS["x008"]["group_width"],
        MODEL_CONFIGS["x008"]["block_type"],
        MODEL_CONFIGS["x008"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX016(model_name="regnetx016",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x016"]["depths"],
        MODEL_CONFIGS["x016"]["widths"],
        MODEL_CONFIGS["x016"]["group_width"],
        MODEL_CONFIGS["x016"]["block_type"],
        MODEL_CONFIGS["x016"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX032(model_name="regnetx032",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x032"]["depths"],
        MODEL_CONFIGS["x032"]["widths"],
        MODEL_CONFIGS["x032"]["group_width"],
        MODEL_CONFIGS["x032"]["block_type"],
        MODEL_CONFIGS["x032"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX040(model_name="regnetx040",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x040"]["depths"],
        MODEL_CONFIGS["x040"]["widths"],
        MODEL_CONFIGS["x040"]["group_width"],
        MODEL_CONFIGS["x040"]["block_type"],
        MODEL_CONFIGS["x040"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX064(model_name="regnetx064",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x064"]["depths"],
        MODEL_CONFIGS["x064"]["widths"],
        MODEL_CONFIGS["x064"]["group_width"],
        MODEL_CONFIGS["x064"]["block_type"],
        MODEL_CONFIGS["x064"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX080(model_name="regnetx080",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x080"]["depths"],
        MODEL_CONFIGS["x080"]["widths"],
        MODEL_CONFIGS["x080"]["group_width"],
        MODEL_CONFIGS["x080"]["block_type"],
        MODEL_CONFIGS["x080"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX120(model_name="regnetx120",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x120"]["depths"],
        MODEL_CONFIGS["x120"]["widths"],
        MODEL_CONFIGS["x120"]["group_width"],
        MODEL_CONFIGS["x120"]["block_type"],
        MODEL_CONFIGS["x120"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX160(model_name="regnetx160",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x160"]["depths"],
        MODEL_CONFIGS["x160"]["widths"],
        MODEL_CONFIGS["x160"]["group_width"],
        MODEL_CONFIGS["x160"]["block_type"],
        MODEL_CONFIGS["x160"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetX320(model_name="regnetx320",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["x320"]["depths"],
        MODEL_CONFIGS["x320"]["widths"],
        MODEL_CONFIGS["x320"]["group_width"],
        MODEL_CONFIGS["x320"]["block_type"],
        MODEL_CONFIGS["x320"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )


## II. RegNetY interfaces : 
def RegNetY002(model_name="regnety002",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y002"]["depths"],
        MODEL_CONFIGS["y002"]["widths"],
        MODEL_CONFIGS["y002"]["group_width"],
        MODEL_CONFIGS["y002"]["block_type"],
        MODEL_CONFIGS["y002"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY004(model_name="regnety004",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y004"]["depths"],
        MODEL_CONFIGS["y004"]["widths"],
        MODEL_CONFIGS["y004"]["group_width"],
        MODEL_CONFIGS["y004"]["block_type"],
        MODEL_CONFIGS["y004"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY006(model_name="regnety006",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y006"]["depths"],
        MODEL_CONFIGS["y006"]["widths"],
        MODEL_CONFIGS["y006"]["group_width"],
        MODEL_CONFIGS["y006"]["block_type"],
        MODEL_CONFIGS["y006"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY008(model_name="regnety008",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y008"]["depths"],
        MODEL_CONFIGS["y008"]["widths"],
        MODEL_CONFIGS["y008"]["group_width"],
        MODEL_CONFIGS["y008"]["block_type"],
        MODEL_CONFIGS["y008"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY016(model_name="regnety016",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y016"]["depths"],
        MODEL_CONFIGS["y016"]["widths"],
        MODEL_CONFIGS["y016"]["group_width"],
        MODEL_CONFIGS["y016"]["block_type"],
        MODEL_CONFIGS["y016"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY032(model_name="regnety032",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y032"]["depths"],
        MODEL_CONFIGS["y032"]["widths"],
        MODEL_CONFIGS["y032"]["group_width"],
        MODEL_CONFIGS["y032"]["block_type"],
        MODEL_CONFIGS["y032"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY040(model_name="regnety040",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y040"]["depths"],
        MODEL_CONFIGS["y040"]["widths"],
        MODEL_CONFIGS["y040"]["group_width"],
        MODEL_CONFIGS["y040"]["block_type"],
        MODEL_CONFIGS["y040"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY064(model_name="regnety064",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y064"]["depths"],
        MODEL_CONFIGS["y064"]["widths"],
        MODEL_CONFIGS["y064"]["group_width"],
        MODEL_CONFIGS["y064"]["block_type"],
        MODEL_CONFIGS["y064"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY080(model_name="regnety080",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y080"]["depths"],
        MODEL_CONFIGS["y080"]["widths"],
        MODEL_CONFIGS["y080"]["group_width"],
        MODEL_CONFIGS["y080"]["block_type"],
        MODEL_CONFIGS["y080"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY120(model_name="regnety120",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y120"]["depths"],
        MODEL_CONFIGS["y120"]["widths"],
        MODEL_CONFIGS["y120"]["group_width"],
        MODEL_CONFIGS["y120"]["block_type"],
        MODEL_CONFIGS["y120"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY160(model_name="regnety160",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y160"]["depths"],
        MODEL_CONFIGS["y160"]["widths"],
        MODEL_CONFIGS["y160"]["group_width"],
        MODEL_CONFIGS["y160"]["block_type"],
        MODEL_CONFIGS["y160"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )

def RegNetY320(model_name="regnety320",
               include_top=True,
               include_preprocessing=True,
               weights="imagenet",
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation="softmax"):
    return RegNet(
        MODEL_CONFIGS["y320"]["depths"],
        MODEL_CONFIGS["y320"]["widths"],
        MODEL_CONFIGS["y320"]["group_width"],
        MODEL_CONFIGS["y320"]["block_type"],
        MODEL_CONFIGS["y320"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    )


if __name__ == "__main__":
    rgy2 = RegNetY002()
    rgy2.summary()