from source.simulator.registry.registry_core import register_task

from source.simulator.scene.track_scene import TrackScene
from source.simulator.dynamics.dynamics.car_dynamics import CarDynamics
from source.tasks.navigation.track_env.mdp.cost import NavigationTrackCost
from source.tasks.navigation.track_env.mdp.termination import TrackTermination
from source.simulator.scene.render import NavigationRenderer


@register_task("navigation_track")
class NavigationTrackTask:

    @staticmethod
    def spec():
        return {
            "scene": TrackScene,
            "dynamics": CarDynamics,
            "cost": NavigationTrackCost,
            "termination": TrackTermination,
            "renderer": NavigationRenderer,
        }