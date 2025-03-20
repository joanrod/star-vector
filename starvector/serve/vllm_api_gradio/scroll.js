var autoScrollBottom = true;

function updateScroll(){
    if (autoScrollBottom) {
        var element = document.getElementsByClassName("cm-scroller")[0];
        element.scrollTop = element.scrollHeight;
    }
}
function handleScroll() {
    var element = document.getElementsByClassName("cm-scroller")[0];
    //if (element.scrollHeight - element.scrollTop === element.clientHeight) {
    if (element.scrollHeight - (element.scrollTop + element.clientHeight) < 0.2*(element.scrollTop)) {
        // User has scrolled to the bottom, enable auto-scrolling
        autoScrollBottom = true;
        console.log("bottom");
    } else {
        console.log("not bottom");
        // User has scrolled away from the bottom, disable auto-scrolling
        autoScrollBottom = false;
    }
}
setInterval(updateScroll, 50);
var element = document.getElementsByClassName("cm-scroller")[0];
element.addEventListener("scroll", handleScroll);
