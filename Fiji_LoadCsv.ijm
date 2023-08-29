// Specify the path to the CSV file
csvPath = File.openDialog("hello");

// Open the CSV file and read its contents
fileContents = File.openAsString(csvPath);
rows = split(fileContents, "\n");

// Loop through the rows of the CSV file
for (i = 0; i < rows.length; i++) {
    // Split the row into columns
    columns = split(rows[i], ",");
    
    // Get the X, Y, Z, and channel coordinates from the first four columns
    x = parseFloat(columns[0]);
    y = parseFloat(columns[1]);
    z = parseInt(columns[2]);
    ch = parseInt(columns[3]);
    
    // Set the current Z-plane and channel
    
    // Create a point selection at the specified coordinates
    makePoint(x, y,"small yellow hybrid");
	Roi.setPosition(z, ch, 1);
	roiManager("Add");
}