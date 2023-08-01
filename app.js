function sendRequest() {
    fetch('http://localhost:5000/api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: 'Hello from client!' })
    })
    .then(response => response.json())
    .then(data => console.log(data));
}
