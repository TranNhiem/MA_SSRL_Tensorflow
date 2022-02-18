## I. Model configuration to load the arch as paper description
MODEL_CONFIGS = {
    "x002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "X"
    },
    "x004": {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "group_width": 16,
        "default_size": 224,
        "block_type": "X"
    },
    "x006": {
        "depths": [1, 3, 5, 7],
        "widths": [48, 96, 240, 528],
        "group_width": 24,
        "default_size": 224,
        "block_type": "X"
    },
    "x008": {
        "depths": [1, 3, 7, 5],
        "widths": [64, 128, 288, 672],
        "group_width": 16,
        "default_size": 224,
        "block_type": "X"
    },
    "x016": {
        "depths": [2, 4, 10, 2],
        "widths": [72, 168, 408, 912],
        "group_width": 24,
        "default_size": 224,
        "block_type": "X"
    },
    "x032": {
        "depths": [2, 6, 15, 2],
        "widths": [96, 192, 432, 1008],
        "group_width": 48,
        "default_size": 224,
        "block_type": "X"
    },
    "x040": {
        "depths": [2, 5, 14, 2],
        "widths": [80, 240, 560, 1360],
        "group_width": 40,
        "default_size": 224,
        "block_type": "X"
    },
    "x064": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 392, 784, 1624],
        "group_width": 56,
        "default_size": 224,
        "block_type": "X"
    },
    "x080": {
        "depths": [2, 5, 15, 1],
        "widths": [80, 240, 720, 1920],
        "group_width": 120,
        "default_size": 224,
        "block_type": "X"
    },
    "x120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224,
        "block_type": "X"
    },
    "x160": {
        "depths": [2, 6, 13, 1],
        "widths": [256, 512, 896, 2048],
        "group_width": 128,
        "default_size": 224,
        "block_type": "X"
    },
    "x320": {
        "depths": [2, 7, 13, 1],
        "widths": [336, 672, 1344, 2520],
        "group_width": 168,
        "default_size": 224,
        "block_type": "X"
    },
    "y002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "Y"
    },
    "y004": {
        "depths": [1, 3, 6, 6],
        "widths": [48, 104, 208, 440],
        "group_width": 8,
        "default_size": 224,
        "block_type": "Y"
    },
    "y006": {
        "depths": [1, 3, 7, 4],
        "widths": [48, 112, 256, 608],
        "group_width": 16,
        "default_size": 224,
        "block_type": "Y"
    },
    "y008": {
        "depths": [1, 3, 8, 2],
        "widths": [64, 128, 320, 768],
        "group_width": 16,
        "default_size": 224,
        "block_type": "Y"
    },
    "y016": {
        "depths": [2, 6, 17, 2],
        "widths": [48, 120, 336, 888],
        "group_width": 24,
        "default_size": 224,
        "block_type": "Y"
    },
    "y032": {
        "depths": [2, 5, 13, 1],
        "widths": [72, 216, 576, 1512],
        "group_width": 24,
        "default_size": 224,
        "block_type": "Y"
    },
    "y040": {
        "depths": [2, 6, 12, 2],
        "widths": [128, 192, 512, 1088],
        "group_width": 64,
        "default_size": 224,
        "block_type": "Y"
    },
    "y064": {
        "depths": [2, 7, 14, 2],
        "widths": [144, 288, 576, 1296],
        "group_width": 72,
        "default_size": 224,
        "block_type": "Y"
    },
    "y080": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 448, 896, 2016],
        "group_width": 56,
        "default_size": 224,
        "block_type": "Y"
    },
    "y120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224,
        "block_type": "Y"
    },
    "y160": {
        "depths": [2, 4, 11, 1],
        "widths": [224, 448, 1232, 3024],
        "group_width": 112,
        "default_size": 224,
        "block_type": "Y"
    },
    "y320": {
        "depths": [2, 5, 12, 1],
        "widths": [232, 696, 1392, 3712],
        "group_width": 232,
        "default_size": 224,
        "block_type": "Y"
    },
}

## II. Weight repo urls 
BASE_WEIGHTS_PATH = "https://storage.googleapis.com/tensorflow/keras-applications/regnet/"

WEIGHTS_HASHES = {
    "x002":
        ("49fb46e56cde07fdaf57bffd851461a86548f6a3a4baef234dd37290b826c0b8",
         "5445b66cd50445eb7ecab094c1e78d4d3d29375439d1a7798861c4af15ffff21"),
    "x004":
        ("3523c7f5ac0dbbcc2fd6d83b3570e7540f7449d3301cc22c29547302114e4088",
         "de139bf07a66c9256f2277bf5c1b6dd2d5a3a891a5f8a925a10c8a0a113fd6f3"),
    "x006":
        ("340216ef334a7bae30daac9f414e693c136fac9ab868704bbfcc9ce6a5ec74bb",
         "a43ec97ad62f86b2a96a783bfdc63a5a54de02eef54f26379ea05e1bf90a9505"),
    "x008":
        ("8f145d6a5fae6da62677bb8d26eb92d0b9dfe143ec1ebf68b24a57ae50a2763d",
         "3c7e4b0917359304dc18e644475c5c1f5e88d795542b676439c4a3acd63b7207"),
    "x016":
        ("31c386f4c7bfef4c021a583099aa79c1b3928057ba1b7d182f174674c5ef3510",
         "1b8e3d545d190271204a7b2165936a227d26b79bb7922bac5ee4d303091bf17a"),
    "x032":
        ("6c025df1409e5ea846375bc9dfa240956cca87ef57384d93fef7d6fa90ca8c7f",
         "9cd4522806c0fcca01b37874188b2bd394d7c419956d77472a4e072b01d99041"),
    "x040":
        ("ba128046c588a26dbd3b3a011b26cb7fa3cf8f269c184c132372cb20b6eb54c1",
         "b4ed0ca0b9a98e789e05000e830403a7ade4d8afa01c73491c44610195198afe"),
    "x064":
        ("0f4489c3cd3ad979bd6b0324213998bcb36dc861d178f977997ebfe53c3ba564",
         "3e706fa416a18dfda14c713423eba8041ae2509db3e0a611d5f599b5268a46c4"),
    "x080":
        ("76320e43272719df648db37271a247c22eb6e810fe469c37a5db7e2cb696d162",
         "7b1ce8e29ceefec10a6569640ee329dba7fbc98b5d0f6346aabade058b66cf29"),
    "x120":
        ("5cafc461b78897d5e4f24e68cb406d18e75f31105ef620e7682b611bb355eb3a",
         "36174ddd0299db04a42631d028abcb1cc7afec2b705e42bd28fcd325e5d596bf"),
    "x160":
        ("8093f57a5824b181fb734ea21ae34b1f7ee42c5298e63cf6d587c290973195d2",
         "9d1485050bdf19531ffa1ed7827c75850e0f2972118a996b91aa9264b088fd43"),
    "x320":
        ("91fb3e6f4e9e44b3687e80977f7f4412ee9937c0c704232664fc83e4322ea01e",
         "9db7eacc37b85c98184070e1a172e6104c00846f44bcd4e727da9e50d9692398"),
    "y002":
        ("1e8091c674532b1a61c04f6393a9c570113e0197f22bd1b98cc4c4fe800c6465",
         "f63221f63d625b8e201221499682587bfe29d33f50a4c4f4d53be00f66c0f12c"),
    "y004":
        ("752fdbad21c78911bf1dcb8c513e5a0e14697b068e5d9e73525dbaa416d18d8e",
         "45e6ba8309a17a77e67afc05228454b2e0ee6be0dae65edc0f31f1da10cc066b"),
    "y006":
        ("98942e07b273da500ff9699a1f88aca78dfad4375faabb0bab784bb0dace80a9",
         "b70261cba4e60013c99d130cc098d2fce629ff978a445663b6fa4f8fc099a2be"),
    "y008":
        ("1b099377cc9a4fb183159a6f9b24bc998e5659d25a449f40c90cbffcbcfdcae4",
         "b11f5432a216ee640fe9be6e32939defa8d08b8d136349bf3690715a98752ca1"),
    "y016":
        ("b7ce1f5e223f0941c960602de922bcf846288ce7a4c33b2a4f2e4ac4b480045b",
         "d7404f50205e82d793e219afb9eb2bfeb781b6b2d316a6128c6d7d7dacab7f57"),
    "y032":
        ("6a6a545cf3549973554c9b94f0cd40e25f229fffb1e7f7ac779a59dcbee612bd",
         "eb3ac1c45ec60f4f031c3f5180573422b1cf7bebc26c004637517372f68f8937"),
    "y040":
        ("98d00118b335162bbffe8f1329e54e5c8e75ee09b2a5414f97b0ddfc56e796f6",
         "b5be2a5e5f072ecdd9c0b8a437cd896df0efa1f6a1f77e41caa8719b7dfcb05d"),
    "y064":
        ("65c948c7a18aaecaad2d1bd4fd978987425604ba6669ef55a1faa0069a2804b7",
         "885c4b7ed7ea339daca7dafa1a62cb7d41b1068897ef90a5a3d71b4a2e2db31a"),
    "y080":
        ("7a2c62da2982e369a4984d3c7c3b32d6f8d3748a71cb37a31156c436c37f3e95",
         "3d119577e1e3bf8d153b895e8ea9e4ec150ff2d92abdca711b6e949c3fd7115d"),
    "y120":
        ("a96ab0d27d3ae35a422ee7df0d789069b3e3217a99334e0ce861a96595bc5986",
         "4a6fa387108380b730b71feea2ad80b5224b5ea9dc21dc156c93fe3c6186485c"),
    "y160":
        ("45067240ffbc7ca2591313fee2f80dbdda6d66ec1a7451446f9a6d00d8f7ac6e",
         "ead1e6b568be8f34447ec8941299a9df4368736ba9a8205de5427fa20a1fb316"),
    "y320": ("b05e173e4ae635cfa22d06392ee3741284d17dadfee68f2aa6fd8cb2b7561112",
             "cad78f74a586e24c61d38be17f3ae53bb9674380174d2585da1a526b8c20e1fd")
}

## III. testing metadata
import numpy as np
# HACKME : if you can find the params table for RegNet 
TEST_REGNET_PARAMS = {
    # RegNet 'X' 
    "RegNetX002" : np.inf,
    "RegNetX004" : np.inf,
    "RegNetX006" : np.inf,
    "RegNetX008" : np.inf,
    "RegNetX016" : np.inf,
    "RegNetX032" : np.inf,
    "RegNetX040" : np.inf,
    "RegNetX064" : np.inf,
    "RegNetX080" : np.inf,
    "RegNetX120" : np.inf,
    "RegNetX160" : np.inf,
    "RegNetX320" : np.inf,
    # RegNet 'Y'
    "RegNetY002" : np.inf,
    "RegNetY004" : np.inf,
    "RegNetY006" : np.inf,
    "RegNetY008" : np.inf,
    "RegNetY016" : np.inf,
    "RegNetY032" : np.inf,
    "RegNetY040" : np.inf,
    "RegNetY064" : np.inf,
    "RegNetY080" : np.inf,
    "RegNetY120" : np.inf,
    "RegNetY160" : np.inf,
    "RegNetY320" : np.inf
}