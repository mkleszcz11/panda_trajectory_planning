from maps.map_config import MapConfig

narrow_passage = MapConfig(
    name='Narrow Passage',
    width=100,  # KEEP IT AS 100
    height=100, # KEEP IT AS 100
    obstacles=[
        (45, 0, 5, 49.5),
        (45, 50.5, 5, 49.5)
    ]
)

def register_map():
    return narrow_passage
