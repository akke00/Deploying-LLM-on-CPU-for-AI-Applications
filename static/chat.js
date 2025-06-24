function renderMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = 'message ' + role;
    msg.innerText = text;
    document.getElementById('messages').appendChild(msg);
    document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text) return;
    renderMessage('user', text);
    input.value = '';

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
    })
    .then(res => res.json())
    .then(data => renderMessage('assistant', data.response));
}

function clearHistory() {
    fetch('/clear', { method: 'POST' })
        .then(() => document.getElementById('messages').innerHTML = '');
}

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => alert(data.status));
}
