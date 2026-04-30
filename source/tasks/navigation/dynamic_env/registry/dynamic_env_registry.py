from source.simulator.registry.registry_core import register_task


from source.simulator.scene.obstacle_scene import DynamicObstacleScene
from source.simulator.dynamics.dynamics.car_dynamics import CarDynamics
from source.tasks.navigation.dynamic_env.mdp.cost import NavigationCost
from source.tasks.navigation.dynamic_env.mdp.termination import NavigationTermination
from source.simulator.scene.render import NavigationRenderer


@register_task("navigation_dynamic")
class NavigationDynamicTask:

    @staticmethod
    def spec():
        return {
            "scene": DynamicObstacleScene,
            "dynamics": CarDynamics,
            "cost": NavigationCost,
            "termination": NavigationTermination,
            "renderer": NavigationRenderer
        }