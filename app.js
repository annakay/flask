function sendRequest() {
    fetch('https://count-the-number-of-people.onrender.com/upload-image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: 'Hello from client!' })
    })
    .then(response => response.json())
    .then(data => console.log(data));
}
