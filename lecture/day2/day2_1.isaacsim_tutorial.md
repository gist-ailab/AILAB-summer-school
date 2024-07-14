## Isaac Sim 용어 정리

- https://docs.omniverse.nvidia.com/isaacsim/latest/reference_glossary.html


### USD

#### USD
- Universal Scene Description (USD)
- Isaac Sim 에서 기본적으로 사용하는 파일 포맷

#### MDL
- Material Definition Language (MDL)
- Omniverse App 들은 MDL 을 이용해서 물체의 Appearance를 표현

  
#### Stage
- 현재 띄워진 Scene의 hierarchy(parent/child)를 보여줌
- https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_stage.html

#### Prim
- primary container object in USD
- USD내의 모든 객체는 `prim` 
- 종류에 따라 `XFormPrim`, `RigidPrim` 등으로 세분화 가능


### IsaacSim CoreAPI (`omni.isaac.core` 안에 있는 `Class` 들)

- https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html


#### World

- `World` is the core class that enables you to interact with the simulator in an easy and modular way
- callbacks, stepping physics, resetting the scene, adding tasks, etc.
- Singleton Class; One `World` per One Isaac Sim

#### Scenes

- `World` 안에서 USD asset 들을 관리하는 Class
- USD asset 을 추가하거나, 조작 하거나 지우는 등의 작업을 수행함

#### Tasks


The Task class in omni.isaac.core provides a way to modularize the scene creation, information retrieval, calculating metrics and creating more complex scenes with more involved logic.


#### Articulations and Robots

An articulated robot is a robot with rotary joints (e.g: a legged robot, a manipulator or a wheeled robot). In omni.isaac.core extension in Omniverse Isaac Sim there exists an Articulation class which enables the interaction with articulations that exists in a USD stage in an easy way.

#### Replicator and Data Generation

Replicator is a Synthetic Data Generation tool for creating parameterizable offline datasets in Omniverse Isaac Sim.

See the omni.replicator extension documentation for additional usage information.



