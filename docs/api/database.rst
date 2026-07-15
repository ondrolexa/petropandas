Database
========

PetroDB is a REST client for the petrodb database, providing access to
projects, samples, spot analyses, and profiles.

PetroDB Client
--------------

.. autoclass:: petropandas._database.PetroDB
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. autoexception:: petropandas._database.PetroDBError

.. autoexception:: petropandas._database.AuthError

.. autoexception:: petropandas._database.NotFoundError

.. autoexception:: petropandas._database.APIError

.. autoexception:: petropandas._database.ReadOnlyError

Data Model
----------

Project
~~~~~~~

.. autoclass:: petropandas._database.Project
   :members:
   :undoc-members:
   :show-inheritance:

Sample
~~~~~~

.. autoclass:: petropandas._database.Sample
   :members:
   :undoc-members:
   :show-inheritance:

Spot
~~~~

.. autoclass:: petropandas._database.Spot
   :members:
   :undoc-members:
   :show-inheritance:

Profile
~~~~~~~

.. autoclass:: petropandas._database.Profile
   :members:
   :undoc-members:
   :show-inheritance:

Area
~~~~

.. autoclass:: petropandas._database.Area
   :members:
   :undoc-members:
   :show-inheritance:

ProfileSpot
~~~~~~~~~~~

.. autoclass:: petropandas._database.ProfileSpot
   :members:
   :undoc-members:
   :show-inheritance:
