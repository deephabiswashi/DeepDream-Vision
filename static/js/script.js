// Function to copy image URL to clipboard
function copyImage(url) {
    navigator.clipboard.writeText(url).then(function() {
        alert("Image URL copied to clipboard!");
    }, function(err) {
        alert("Failed to copy URL: " + err);
    });
}

// Theme toggling functionality
function toggleTheme() {
    let currentTheme = localStorage.getItem('theme') || 'vibrant';
    let newTheme = (currentTheme === 'vibrant') ? 'dark' : 'vibrant';
    document.body.classList.remove(currentTheme + '-theme');
    document.body.classList.add(newTheme + '-theme');
    localStorage.setItem('theme', newTheme);
}

// Set theme on initial load
document.addEventListener("DOMContentLoaded", function() {
    let currentTheme = localStorage.getItem('theme') || 'vibrant';
    document.body.classList.add(currentTheme + '-theme');
});
