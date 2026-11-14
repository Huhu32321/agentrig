"""Benchmark cases: permissively licensed, artist-rigged humanoids (pinned commits).

The models are NOT redistributed with this repository; `download.py` fetches them.
"""
QUAT = ("https://raw.githubusercontent.com/Yametech12/ghostname-3d-assets/"
        "3ba341c4346e7833da316f4f8b127d349ca89041/models/{}.glb")
KAY = ("https://raw.githubusercontent.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0/"
       "672074b73ba276876a19e8816ecdc5241817ab47/addons/kaykit_character_pack_adventures/"
       "Characters/gltf/{}.glb")

QUAT_NAMES = ["Casual_Male", "Casual_Female", "Casual2_Male", "Casual2_Female", "Casual3_Male",
              "Casual_Bald", "Doctor_Male_Young", "Doctor_Female_Young", "OldClassy_Male",
              "OldClassy_Female", "Suit_Male", "Suit_Female", "Worker_Male", "Worker_Female"]
KAY_NAMES = ["Barbarian", "Knight", "Mage", "Rogue", "Rogue_Hooded"]

CASES = (
    [{"id": f"quat_{n.lower()}", "url": QUAT.format(n), "style": "stylised-realistic",
      "credit": "Quaternius, Ultimate Animated Character Pack", "license": "CC0-1.0"} for n in QUAT_NAMES]
    + [{"id": f"kay_{n.lower()}", "url": KAY.format(n), "style": "chibi",
        "credit": "Kay Lousberg, KayKit Adventurers 1.0", "license": "CC0-1.0"} for n in KAY_NAMES]
    + [{"id": "cesium_man", "style": "realistic-lowpoly",
        "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
               "8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/CesiumMan/glTF-Binary/CesiumMan.glb",
        "credit": "Cesium (Khronos glTF Sample Assets)", "license": "CC-BY-4.0"},
       {"id": "robot_expressive", "style": "cartoon-robot",
        "url": "https://raw.githubusercontent.com/mrdoob/three.js/"
               "253fa7ad003181df28015efdb541c2916895a816/examples/models/gltf/RobotExpressive/RobotExpressive.glb",
        "credit": "Tomas Laulhe (Quaternius), three.js examples", "license": "CC0-1.0"}]
)
