import csv
import overpass

def get_relation_ways(relation_id):
    # Create the Overpass API query string
    sQuery = f"relation({relation_id});(._;>;);"

    # Create the Overpass API object
    oAPI = overpass.API()

    # Posts the query to the Overpass API
    oAPIResponse = oAPI.get(sQuery, responseformat="geojson", verbosity="body ids tags geom", build=True)

    # Get the coordinates from the query response that is a dictionary of dictionaries
    oElementIDs = []
    for oElement in oAPIResponse["features"]:
        nNodeID = oElement["id"]
        if nNodeID == 4937755860:
            print("here")
        oElementIDs.append(nNodeID)

    return oElementIDs


def get_road_coordinates(id):
    # Create the Overpass API query string
    sQuery = f"way({id});"

    # Create the Overpass API object
    oAPI = overpass.API()

    # Posts the query to the Overpass API
    oAPIResponse = oAPI.get(sQuery, responseformat="geojson", verbosity="ids geom", build=True)

    # Get the coordinates from the query response that is a dictionary of dictionaries
    oRoadCoordinates = []
    for oElement in oAPIResponse["features"]:
        assert oElement["id"] == id, "wrong ID retrieved"
        for oCoord in oElement["geometry"]["coordinates"]:
            oRoadCoordinates.append(oCoord)

    return oRoadCoordinates


CIRCUIT_DE_MONACO_OSM_ID = 148194
oWayIDs = get_relation_ways(CIRCUIT_DE_MONACO_OSM_ID)

all_coordinates = []
for nID in oWayIDs:
    oRoadCoordinates = get_road_coordinates(nID)
    all_coordinates.extend(oRoadCoordinates)

# Specify the CSV file path
csv_file = "monaco_coordinates.csv"

# Write coordinates to CSV file
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Latitude", "Longitude"])
    writer.writerows(all_coordinates)

print(f"Coordinates saved to {csv_file}.")
