import csv
import overpass

def get_relation_ways(relation_id):
    # Create the Overpass API query string
    sQuery = f"relation({relation_id});(._;>;);"

    # Create the Overpass API object
    oAPI = overpass.API()

    # Post the query to the Overpass API
    oAPIResponse = oAPI.get(sQuery, responseformat="geojson", verbosity="body")

    # Get the ways from the query response
    ways = oAPIResponse["features"]
    return ways


def get_way_coordinates(way):
    # Get the coordinates from the way geometry
    oRoadCoordinates = []
    geometry = way["geometry"]
    if geometry["type"] == "Point":
        oRoadCoordinates.append(geometry["coordinates"])
    elif geometry["type"] == "LineString":
        oRoadCoordinates.extend(geometry["coordinates"])

    return oRoadCoordinates


if __name__ == "__main__":
    CIRCUIT_DE_MONACO_OSM_ID = 148194

    # Get the ways for the Monaco track
    ways = get_relation_ways(CIRCUIT_DE_MONACO_OSM_ID)

    # Extract the coordinates from the ways
    track_coordinates = []
    for way in ways:
        way_coordinates = get_way_coordinates(way)
        track_coordinates.extend(way_coordinates)

    # Specify the CSV file path
    csv_file = "monaco_track_coordinates.csv"

    # Write coordinates to CSV file
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Latitude", "Longitude"])
        for coordinate in track_coordinates:
            writer.writerow([coordinate[1], coordinate[0]])

    print(f"Track coordinates saved to {csv_file}.")
