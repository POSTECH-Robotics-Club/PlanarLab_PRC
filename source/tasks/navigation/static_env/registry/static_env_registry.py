from source.simulator.registry.registry_core import register_task


from source.simulator.scene.obstacle_scene import ObstacleScene
from source.simulator.dynamics.dynamics.car_dynamics import CarDynamics
from source.tasks.navigation.static_env.mdp.cost import NavigationCost
from source.tasks.navigation.static_env.mdp.termination import NavigationTermination
from source.simulator.scene.render import NavigationRenderer


@register_task("navigation_static")
class NavigationStaticTask:

    @staticmethod
    def spec():
        return {
            "scene": ObstacleScene,
            "dynamics": CarDynamics,
            "cost": NavigationCost,
            "termination": NavigationTermination,
            "renderer": NavigationRenderer
        }