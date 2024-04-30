
function uploadImage() {
  var fileInput = document.getElementById('imageInput');
  var output = document.getElementById('imageDisplay');

  if (fileInput.files.length > 0) {
    var selectedFile = fileInput.files[0];
    var image = new Image();

    image.onload = function() {
      // display the uploaded image
      output.innerHTML = '';
      output.appendChild(image);

      // show the message of uploaded successfully
      var uploadedMessage = document.createElement('p');
      uploadedMessage.textContent = 'Image uploaded successfully!';
      output.appendChild(uploadedMessage);

      // show the delete button
      document.getElementById('deleteButton').style.display = 'block';
    };

    image.src = URL.createObjectURL(selectedFile);
    image.alt = 'Uploaded Image';

    // clean the input
    fileInput.value = '';
  } else {
    alert('Please select an image file before uploading.');
  }
}

// Click the Delete button to empty the image container
function deleteImage() {
  var output = document.getElementById('imageDisplay');
  output.innerHTML = '';

  // hide the delete button
  document.getElementById('deleteButton').style.display = 'none';
}


//click the predict button to link the back-end
function predictImage() {
  var output = document.getElementById('imageDisplay');

  if (output.firstChild) {
    var formData = new FormData();
    var imageFile = output.firstChild.src;

    // Create a Blob object and convert the URL of the image to a Blob
    fetch(imageFile)
      .then(response => response.blob())
      .then(blob => {
        // Add Blob objects to FormData objects
        formData.append('imageFile', blob, 'image.png');

        // send the request to the back-end to predict
        fetch('/predict', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          // display the prediction result
          resultText.textContent = data.prediction;
        })
        .catch(error => {
          console.error('Error:', error);
        });
      })
      .catch(error => {
        console.error('Error fetching image:', error);
      });
  } else {
    alert('Please upload an image before predicting.');
  }
}

