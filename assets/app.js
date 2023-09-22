const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('show');
        } else {
            entry.target.classList.remove('show');
        }
    });
});

const hiddenElementsJa = document.querySelectorAll('.hidden-ja');  
hiddenElementsJa.forEach(element => {
    observer.observe(element);
});

const hiddenElementsEn = document.querySelectorAll('.hidden-en');  
hiddenElementsEn.forEach(element => {
    observer.observe(element);
});

document.addEventListener('DOMContentLoaded', () => {
    const players = Array.from(document.querySelectorAll('.player')).map(p => new Plyr(p));
});

function handleButtonClick(event, type) {
    const jaElements = document.querySelectorAll('.ja .text'); // Get all Japanese text elements
    const buttons = document.querySelectorAll('.bar-button'); // Get all buttons

    // Remove the active-button class from all buttons
    buttons.forEach(button => {
        button.classList.remove('active-button');
    });

    jaElements.forEach(element => {
        let blockData;
        
        // Depending on the button pressed, get the corresponding data attribute
        switch (type) {
            case 'Off':
                blockData = element.getAttribute('data-ja_no_furi');
                break;
            case 'First':
                blockData = element.getAttribute('data-ja_first_furi');
                break;
            case 'On':
                blockData = element.getAttribute('data-ja_all_furi');
                break;
        }

        // If blockData exists, set the innerHTML content
        if (blockData) {
            element.innerHTML = blockData;
        }
    });

    // Add the active-button class to the clicked button
    event.target.classList.add('active-button');
}